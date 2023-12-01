# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:27:49 2023

@author: Admin
"""
# Import the necessary packages 
from imutils import face_utils 
# import dlib
import cv2 
from scipy.spatial import distance as dist

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear



def find_eye_status(image,detector,predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces 
    rects = detector(image, 1)
    status='active'
    if len(rects)==1:
        
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            # Convert it to a (68, 2) size numpy array 
            shape = face_utils.shape_to_np(shape)
            
            leftEye = shape[lstart:lend]
            rightEye = shape[rstart:rend] 
            # Compute the EAR for both the eyes 
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
    
            # Take the average of both the EAR
            EAR = (leftEAR + rightEAR) / 2.0
            #live datawrite in csv
    
            if EAR < 0.2: 
                status='sleepy'
    return status

# Grab the indexes of the facial landamarks for the left and right eye respectively 
(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# image = cv2.imread("C:/Users/Admin/Downloads/backup1/Eye_movement_tracking/sample/without_mask_1 (37).jpg")
# print(find_eye_status(image))