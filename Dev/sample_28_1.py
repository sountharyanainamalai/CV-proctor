# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:06:44 2023

@author: Admin
"""

import cv2
import numpy as np
import dlib
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
dlibFaceDetector = dlib.get_frontal_face_detector()

def face_detection(image):
    dlib_rect = dlibFaceDetector(image,1)   
    gray_img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haar_detector = face_cascade.detectMultiScale(gray_img)
    print(len(haar_detector),len(dlib_rect))
    if len(dlib_rect)==1 and len(haar_detector)==0:
        print("face direction not towards camera")
    elif len(dlib_rect)==1 and len(haar_detector)==1:
        eyes= eye_cascade.detectMultiScale(gray_img,1.3,5)
        # print(len(eyes))
        if len(eyes)==2:
            if eyes[0][0]<eyes[1][0]:
                cropped_img=image[eyes[0][1]:(eyes[1][1]+eyes[1][3]),eyes[0][0]:(eyes[1][0]+eyes[1][2])]
                cv2.imwrite('only_eye.png',cropped_img)
                height, width,c = cropped_img.shape
                # Cut the image in half
                width_cutoff = width // 2
                left_eye = cropped_img[:, :width_cutoff]
                right_eye = cropped_img[:, width_cutoff:]
                cv2.imwrite('eye.png',left_eye)
                height, width = left_eye.shape[:2]
                cut_eye_brow = left_eye[5:height, 0:width]
                cv2.imwrite('eye.png',cut_eye_brow)
                gray = cv2.cvtColor(cut_eye_brow, cv2.COLOR_BGR2GRAY)

                # _, img = cv2.threshold(cut_eye_brow, 42, 255, cv2.THRESH_BINARY)
                # img = cv2.erode(img, None, iterations=2)
                # img = cv2.dilate(img, None, iterations=4)
                # img = cv2.medianBlur(img, 5)
                _, eye_thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(eye_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                   pupil = max(contours, key=cv2.contourArea)
                   x1, y1, w1, h1 = cv2.boundingRect(pupil)
                   center = (int(x1 + w1/2), int(y1 + h1/2))
                   cv2.circle(cut_eye_brow, center, 3, (255, 0, 0),-1)
                cv2.imwrite('eye.png',cut_eye_brow)

                
            else:
                cropped_img=image[eyes[1][1]-40:(eyes[0][1]+eyes[0][3]),eyes[1][0]-5:(eyes[0][0]+eyes[0][2])+10]
                cv2.imwrite('eye1.png',cropped_img)
            print("eye test")
            

    elif len(dlib_rect)>1:
        print("multiple face detected in one frame")
    else:
        print("no face detected")



if __name__=='__main__':
    image=cv2.imread('C:/Users/Admin/Downloads/H040011-20231128T115049Z-001/H040011/H040011_20191127194108_Training.jpg')
    face_detection(image)

