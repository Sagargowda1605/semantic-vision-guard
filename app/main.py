import tempfile

from fastapi import FastAPI,UploadFile,File
from typing import List
import cv2 as cv
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
import os
from PIL import Image
from app.database import init_db,log_detection
from app.models.clip import ClipScapper
from app.models.opencv import extract_rois,show_rois
from app.models.yolo import COCO_CLASSES,detect_yolo
from app.models.yolo_onxx import detect_yolo_onnx,postprocess


init_db()  # Initialize the database when the app starts

app=FastAPI(title="Vision Guard API")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Join it with "static" to point exactly to your static folder
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Mount the static directory using the absolute path
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

clip=ClipScapper()

text_cache={}
for cls_id,cls_name in COCO_CLASSES.items():
    clip.set_prompt(cls_name)
    text_cache[cls_id]=clip._text_feat

clip_threshold=0.20

@app.get("/")
def home():
    # This assumes your HTML file is named 'index.html' and is inside the 'static' folder
    ui_path = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(ui_path)

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

            try:
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
            finally:
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
            log_detection(
                filename="temp_file",
                yolo_class=COCO_CLASSES.get(cls_id,"unknown"),
                yolo_conf=conf,
                clip_score=sim,
                status="detected",
                bbox=[int(x1),int(y1),int(x2),int(y2)]
            )
    return result
