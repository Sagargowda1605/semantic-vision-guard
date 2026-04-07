import cv2 as cv 
import os
from typing import List,Tuple
from PIL import Image
from dataclasses import dataclass




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

    
        x1, y1, x2, y2 = boxes[i]
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



 


 

 