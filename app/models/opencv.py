import cv2 as cv 
import os
from yolo import detect_yolo,COCO_CLASSES
from typing import List,Tuple
from PIL import Image
from dataclasses import dataclass
from clip import ClipScapper


clip=ClipScapper()

text_cache={}
for cls_id,cls_name in COCO_CLASSES.items():
    clip.set_prompt(cls_name)
    text_cache[cls_id]=clip._text_feat

print('done caching text features')

clip_threshold=0.20

"""Think of a decorator like gift wrapping: it wraps your original function or class in another layer of code, 
allowing you to add new behavior without permanently changing the source code of the original function."""
@dataclass #decorater
class DetectorROI:
    crop_bgr:"cv.Mat"
    box_xyxy:Tuple[int,int,int,int]
    cls_id:int
    conf:float

def extract_rois(boxes,frame,clss,confs):
    h,w,_=frame.shape
    rois:List[DetectorROI]=[]
     

    for i in range(len(boxes)):

    
        x1, y1, x2, y2 = boxes[i].tolist()
        x1=max(0,int(x1))
        x2=min(w,int(x2))
        y1=max(0,int(y1))
        y2=min(h,int(y2))

        if x2<=x1 or y2<=y1:
            print("failed")
            continue 

        crop_photo=frame[y1:y2,x1:x2]
        rois.append(
            DetectorROI(
                crop_bgr=crop_photo,
                box_xyxy=(x1, y1, x2, y2),
                cls_id=int(clss[i]),
                conf=float(confs[i]),
            )
        )
    return rois

def show_rois(rois:List[DetectorROI]):

    for roi in rois:
        img=roi.crop_bgr.copy()
        cv.putText(img,f"Label data Car",(5,20),cv.FONT_HERSHEY_SIMPLEX,0.2,(0,255,0),2)
        cv.imshow("cropped",img)
        key=cv.waitKey(0)
        if key==27:
            break



def run_video(video_path:str):
    cap=cv.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"File is not found at {video_path}")
    
    frame_idx=0

    while True:
        ok,frame=cap.read()
        if not ok :
            break
        r0=detect_yolo(frame)
        boxes=r0.boxes.xyxy
        confs=r0.boxes.conf
        clss=r0.boxes.cls
        rois=extract_rois(boxes,frame,clss,confs)

        for roi in rois:
            crop=roi.crop_bgr
            cls_id=roi.cls_id
            conf=roi.conf

            
            clip._text_feat = text_cache.get(cls_id)
            if clip._text_feat is None:
                continue
            img_pil=Image.fromarray(cv.cvtColor(crop,cv.COLOR_BGR2RGB))
            sim=clip.score_pill(img_pil)
            #print(f"{COCO_CLASSES[cls_id]}: CLIP score = {sim:.3f}")

            if sim>=clip_threshold:
                x1,y1,x2,y2=roi.box_xyxy
                cv.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv.putText(frame,f"{COCO_CLASSES[cls_id]} {sim:.2f}",(x1,y1-10),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        cv.imshow("frame",frame)

        if cv.waitKey(1) & 0xff==ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()



def main():
    run_video("data/input/Snow_Video.mp4")

if __name__ == "__main__":
    main()


 

 