import cv2
import dlib
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
dlibFaceDetector = dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    dlib_rect = dlibFaceDetector(frame,0)   
    gray_img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    haar_detector = face_cascade.detectMultiScale(gray_img)
    print(len(haar_detector),len(dlib_rect))
    if len(dlib_rect)==1 and len(haar_detector)==0:
        print("face direction not towards camera")
    elif len(dlib_rect)==1 and len(haar_detector)>=1:
        shape=predictor(frame,dlib_rect[0])
        # 37 40
        # a=shape.part(18).x
        # b=shape.part(32).x 
        # c=shape.part(21).y 
        # d=shape.part(28).y
        
        x1=shape.part(18).x
        x2=shape.part(32).x 
        y1=shape.part(20).y 
        y2=shape.part(28).y
        
        left_eye=frame[y1:y2,x1:x2]
        gray = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
        _, eye_thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        # cv2.imwrite('C:/Users/Admin/Downloads/backup1/Eye_movement_tracking/thresh/'+'eyethresh_'+str(name.split('.')[0])+'.png',eye_thresh)
        contours, _ = cv2.findContours(eye_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(i)
        if len(contours) > 0:
            pupil = max(contours, key=cv2.contourArea)
            x1, y1, w1, h1 = cv2.boundingRect(pupil)
            center = (int(x1 + w1/2), int(y1 + h1/2))
            cv2.circle(left_eye, center, 2, (255, 0, 0),-1)
        # cv2.imwrite('C:/Users/Admin/Downloads/backup1/Eye_movement_tracking/result/'+'res_eye_'+str(name.split('.')[0])+'.png',left_eye)


    elif len(dlib_rect)>1:
        print("multiple face detected in one frame")
    else:
        print("no face detected")
        
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # faces = face_detector(gray)

    # for face in faces:
    #     x, y, w, h = face.left(), face.top(), face.width(), face.height()

    #     roi_left_eye = gray[y + int(0.1 * h):y + int(0.4 * h), x + int(0.1 * w):x + int(0.5 * w)]
    #     roi_right_eye = gray[y + int(0.1 * h):y + int(0.4 * h), x + int(0.5 * w): x + int(0.9 * w)]

    #     _, thresh_left_eye = cv2.threshold(roi_left_eye, 30, 255, cv2.THRESH_BINARY)
    #     _, thresh_right_eye = cv2.threshold(roi_right_eye, 30, 255, cv2.THRESH_BINARY)
        
    #     contours_left, _ = cv2.findContours(thresh_left_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #     contours_right, _ = cv2.findContours(thresh_right_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
    #     for contour in contours_left:
    #         x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
    #         center_x = x + int(0.1 * w) + x_c + w_c // 2 
    #         center_y = y + int(0.1 * h) + y_c + h_c // 2 
    #         radius = max(w_c, h_c) // 3 
    #         cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 2)            

    #     for contour in contours_right:
    #         x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
    #         center_x = x + int(0.5 * w) + x_c + w_c // 2 
    #         center_y = y + int(0.1 * h) + y_c + h_c // 2 
    #         radius = max(w_c, h_c) // 3
    #         cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 2)


    cv2.imshow('Eye Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()