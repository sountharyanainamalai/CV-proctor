import cv2
import dlib
import numpy as np
import os
from natsort import natsorted
def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    
    l = points[0][0]
    t = (points[1][1]+points[2][1])//2
    r = points[3][0]
    b = (points[4][1]+points[5][1])//2
    return mask, [l, t, r, b]

def find_eyeball_position(cen, s, d):
    """Find and return the eyeball positions, i.e. left or right or top or normal"""
    if cen<100 and s>cen and d>cen:
            pos=0
    elif cen<100 or s<cen or d<cen:
            if s<cen and s<d and abs(cen-s)>10:
                pos=1
            elif d<cen and d<s and abs(cen-d)>10:
                pos=2
            else:
                pos=0
    elif cen>=100:
        if s<cen and d<cen:
            pos=0
        if s>cen and d<cen:
            pos=1
        elif s<cen and d>cen:
            pos=2
        else:
            pos=0
    return pos
    
def contouring(thresh, mid, img, end_points, gray,right=False):
    """
    Find the largest contour on an image divided by a midpoint and subsequently the eye position

    Parameters
    ----------
    thresh : Array of uint8
        Thresholded image of one side containing the eyeball
    mid : int
        The mid point between the eyes
    img : Array of uint8
        Original Image
    end_points : list
        List containing the exteme points of eye
    right : boolean, optional
        Whether calculating for right eye or left eye. The default is False.

    Returns
    -------
    pos: int
        the position where eyeball is:
            0 for normal
            1 for left
            2 for right
            3 for up

    """
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        x1, y1, w1, h1 = cv2.boundingRect(cnt)
        eye_crop=gray[y1:y1+h1,x1:x1+w1]
        h,w=eye_crop.shape
        first_part=eye_crop[:,:w//2]
        sec_part=eye_crop[:,w//2:]
        # cv2.imwrite('on.png',eye_crop)
        # cv2.circle(eye_crop, (w//2,h//2), 2, (255, 0, 0),-1)
        mid_val=eye_crop[h//2,w//2]
        left_val=first_part[first_part.shape[0]//2,first_part.shape[1]//2]
        right_val=sec_part[sec_part.shape[0]//2,sec_part.shape[1]//2]

        # gray[y1:y1+h1,x1:x1+w1]=eye_crop
        # cv2.rectangle(img, (x1,y1),(x1+w1,y1+h1),(255,0,0))
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        pos = find_eyeball_position(mid_val, left_val, right_val)
        return pos
    except:
        pass
    
def process_thresh(thresh):
    """
    Preprocessing the thresholded image

    Parameters
    ----------
    thresh : Array of uint8
        Thresholded image to preprocess

    Returns
    -------
    thresh : Array of uint8
        Processed thresholded image

    """
    thresh = cv2.erode(thresh, None, iterations=4)
    thresh = cv2.dilate(thresh, None, iterations=4) 
    thresh = cv2.medianBlur(thresh, 3) 
    thresh = cv2.bitwise_not(thresh)
    return thresh

def print_eye_pos(img, left, right):
    """
    Print the side where eye is looking and display on image

    Parameters
    ----------
    img : Array of uint8
        Image to display on
    left : int
        Position obtained of left eye.
    right : int
        Position obtained of right eye.

    Returns
    -------
    None.

    """
    if left == right and left != 0:
        text = ''
        if left == 1:
            print('Looking right')
            text = 'Looking right'
        elif left == 2:
            print('Looking left')
            text = 'Looking left'
        elif left == 3:
            print('Looking up')
            text = 'Looking up'
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, text, (30, 30), font,  
                   1, (0, 255, 255), 2, cv2.LINE_AA) 

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:/Users/Admin/Downloads/backup1/Eye_movement_tracking/shape_predictor_68_face_landmarks.dat')

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0)
ret, img = cap.read()
thresh = img.copy()

while(True):
    ret, img = cap.read()
    
    kernel = np.ones((9, 9), np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for rect in rects:
    
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask,end_points_left = eye_on_mask(mask, left)
        mask,end_points_right = eye_on_mask(mask, right)
        mask = cv2.dilate(mask, kernel, 5)
        eyes = cv2.bitwise_and(img, img, mask=mask)
        # cv2.imwrite('C:/Users/Admin/Downloads/backup1/Eye_movement_tracking/eye_images/'+'eyes'+str(i.split('.')[0])+'.png',eyes)

        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        # cv2.imwrite('C:/Users/Admin/Downloads/backup1/Eye_movement_tracking/thresh/'+'eyes'+str(i.split('.')[0])+'.png',eyes)

        mid = int((shape[42][0] + shape[39][0]) // 2)
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        # threshold = cv2.getTrackbarPos('threshold', 'image')
        _, thresh = cv2.threshold(eyes_gray, 75, 255, cv2.THRESH_BINARY)
        # cv2.imwrite('C:/Users/Admin/Downloads/backup1/Eye_movement_tracking/thresh/'+'eyes'+str(i.split('.')[0])+'.png',thresh)

        thresh = process_thresh(thresh)
        # cv2.imwrite('C:/Users/Admin/Downloads/backup1/Eye_movement_tracking/thresh/'+'eyes'+str(i.split('.')[0])+'.png',thresh)

        eyeball_pos_left = contouring(thresh[:, 0:mid], mid, img, end_points_left,gray[:, 0:mid])
        eyeball_pos_right = contouring(thresh[:, mid:], mid, img, end_points_right,gray[:, mid:], True)
        # print(eyeball_pos_left, eyeball_pos_right)
        print_eye_pos(img, eyeball_pos_left, eyeball_pos_right)
        # cv2.imwrite('C:/Users/Admin/Downloads/backup1/Eye_movement_tracking/result/'+'org_img'+str(i.split('.')[0])+'.png',img)
    cv2.imshow('eyes', img)
    cv2.imshow("image", thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
