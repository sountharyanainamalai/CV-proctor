# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 17:18:58 2023

@author: Admin
"""

from yolo_detect_and_count import YOLOv8_ObjectDetector
import cv2
def img_bound_box(box,img):
               # Bounding box
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
 
    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
    org = [x1, y1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
 
    cv2.putText(img, class_name, org, font, fontScale, color, thickness)

headphone_model = 'C:/Users/Admin/Downloads/best.pt'
headphone_detector = YOLOv8_ObjectDetector(headphone_model, conf = 0.40 )
im='C:/Users/Admin/Downloads/earphones.jpg'
im_read=cv2.imread(im)
results=headphone_detector.predict_img(im)
classNames = ["headphone"]
for r in results:
    boxes = r.boxes
    for box in boxes:
        # Class name
        cls = int(box.cls[0])
        # print(cls)
        class_name = classNames[cls]
        img_bound_box(box,im_read)
        
cv2.imwrite('res.png',im_read)
