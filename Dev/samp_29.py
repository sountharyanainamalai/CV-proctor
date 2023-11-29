# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:50:30 2023

@author: Admin
"""

import cv2
import numpy as np
import dlib
from PIL import Image
import os
IMAGE_SIZE = 1800

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
dlibFaceDetector = dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def contrast_enhancement(img):
    # img = cv2.imread(img_path)
    #show_grayscale_histogram(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    """ Splitting LAB image to different Channel """
    #cv2.imwrite("newL.jpg",l)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    """ merge the CLAHE enhanced L channel with the original A and B channel"""
    limg = cv2.merge((cl,a,b))
    
    #cv2.imwrite('limg.jpg', limg)
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    #plt.hist(final.flat, bins = 100, range = (0,256))
    #plt.show()
    #cv2.imwrite("%s" %(img_path), final)
    cv2.imwrite("Image_final.jpg", final)
    return final

def image_scaling(img_path):
    """ Scanning at 300 dpi (dots per inch) is not officially a standard for OCR (optical character recognition), 
    but it is considered the gold standard.
    Tesseract works well on Image which have at least 300 DPIs"""
    im = Image.open(img_path)
    len_x, wid_y = im.size
    factor = max(1, int(IMAGE_SIZE / len_x))
    newsize = factor * len_x, factor * wid_y
    img_resize = im.resize(newsize, Image.ANTIALIAS)
    img_resize.save(img_path, dpi = (300,300))
    
    
    

def face_detection(image,name):
    dlib_rect = dlibFaceDetector(image,1)   
    gray_img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haar_detector = face_cascade.detectMultiScale(gray_img)
    print(len(haar_detector),len(dlib_rect))
    if len(dlib_rect)==1 and len(haar_detector)==0:
        print("face direction not towards camera")
    elif len(dlib_rect)==1 and len(haar_detector)>=1:
        shape=predictor(image,dlib_rect[0])
        # 37 40
        # a=shape.part(18).x
        # b=shape.part(32).x 
        # c=shape.part(21).y 
        # d=shape.part(28).y
        
        x1=shape.part(18).x
        x2=shape.part(32).x 
        y1=shape.part(20).y 
        y2=shape.part(28).y
        
        left_eye=image[y1:y2,x1:x2]
        gray = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('C:/Users/Admin/Downloads/backup1/Eye_movement_tracking/eye_images/'+'only_eyes_'+str(name.split('.')[0])+'.png',left_eye)
        _, eye_thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite('C:/Users/Admin/Downloads/backup1/Eye_movement_tracking/thresh/'+'eyethresh_'+str(name.split('.')[0])+'.png',eye_thresh)
        contours, _ = cv2.findContours(eye_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(i)
        if len(contours) > 0:
            pupil = max(contours, key=cv2.contourArea)
            x1, y1, w1, h1 = cv2.boundingRect(pupil)
            center = (int(x1 + w1/2), int(y1 + h1/2))
            cv2.circle(left_eye, center, 2, (255, 0, 0),-1)
        cv2.imwrite('C:/Users/Admin/Downloads/backup1/Eye_movement_tracking/result/'+'res_eye_'+str(name.split('.')[0])+'.png',left_eye)


    elif len(dlib_rect)>1:
        print("multiple face detected in one frame")
    else:
        print("no face detected")



if __name__=='__main__':
    # image=cv2.imread('C:/Users/Admin/Downloads/H040011-20231128T115049Z-001/H040011/H040011_20191127194108_Training.jpg')
    main_path='C:/Users/Admin/Downloads/backup1/Eye_movement_tracking/sample'
    listDir=os.listdir(main_path)
    for i in listDir[0:10]:
        image=cv2.imread(os.path.join(main_path,i))
        face_detection(image,i)

