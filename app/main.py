import tempfile

from fastapi import FastAPI,UploadFile,File
from typing import List
import cv2 as cv
from fastapi.responses import JSONResponse
import numpy as np
import os
from PIL import Image
from app.models.clip import ClipScapper
from app.models.opencv import extract_rois,show_rois
from app.models.yolo import COCO_CLASSES,detect_yolo
from app.models.yolo_onxx import detect_yolo_onnx,postprocess

app=FastAPI(title="Vision Guard API")

clip=ClipScapper()

text_cache={}
for cls_id,cls_name in COCO_CLASSES.items():
    clip.set_prompt(cls_name)
    text_cache[cls_id]=clip._text_feat

clip_threshold=0.20

@app.get("/")
def home():
    return {"message":"Welcome to Vision Guard API"}

@app.post("/detect/")
async def detect(file:UploadFile=File(...)):

    suffix=os.path.splitext(file.filename)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path=tmp.name
    detections=[]

    try:
        if suffix in [".jpg",".jpeg",".png"]:
            frame=cv.imread(tmp_path)
            detections=process_frames(frame)
        elif suffix in [".mp4",".avi",".mov"]:
            cap=cv.VideoCapture(tmp_path)
            frame_count=0

            while frame_count<30: # Process first 30 frames for demo
                ret,frame=cap.read()
                if not ret:
                    break
                detections.extend(process_frames(frame))
                frame_count+=1
            
                cv.imshow("frame",frame)

                if cv.waitKey(1) & 0xff==ord("q"):
                    break
        cap.release()
        cv.destroyAllWindows()
             
    finally:
        os.unlink(tmp_path)
    
    return JSONResponse(content={"filename": file.filename,"total_detections": len(detections),"detections": detections})

def process_frames(frame):

    result=[]
    r0=detect_yolo_onnx(frame)
    final_boxes, final_confs, final_cls = postprocess(r0, frame, conf_threshold=0.25, iou_threshold=0.5)
    x1y1x2y2=final_boxes
    confs=final_confs
    clss=final_cls
    rois=extract_rois(x1y1x2y2,frame,clss,confs)
    for roi in rois:
        crop=roi.crop_bgr
        cls_id=roi.cls_id
        conf=roi.conf

            
        clip._text_feat = text_cache.get(cls_id)
        if clip._text_feat is None:
            continue
        img_pil=Image.fromarray(cv.cvtColor(crop,cv.COLOR_BGR2RGB))
        sim=clip.score_pill(img_pil)

        if sim>=clip_threshold:
            x1,y1,x2,y2=roi.box_xyxy
            result.append({
                "class": COCO_CLASSES.get(cls_id,"unknown"),
                "confidence": round(float(conf),3),
                "clip_score": round(float(sim),3),
                "box": [int(x1),int(y1),int(x2),int(y2)]
            })
    return result
