import cv2
from face_detector import get_face_detector, find_faces
import os
from model_arch import Model
import torchvision.transforms as transforms
import torch
from PIL import Image


print("[INFO] loading liveness detector...")
model = Model()
model.load_state_dict(torch.load(os.path.join(os.getcwd(), "model.pth"),map_location='cpu'))
# model.cuda()
model.eval()

face_model = get_face_detector()
cap = cv2.VideoCapture(0)



while True:
    ret, img = cap.read()
    # img = imutils.resize(img, width=600)

    (h, w) = img.shape[:2]

    face_bboxes = find_faces(img, face_model)
    height, width = img.shape[:2]
    if len(face_bboxes)==1:
        (startX, startY, endX, endY) =face_bboxes[0]
        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(w, endX)
        endY = min(h, endY)

			# extract the face ROI and then preproces it in the exact
			# same manner as our training data
        face = img[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = Image.fromarray(face)
        face = face.transpose(Image.ROTATE_270)
        transform = transforms.Compose(
				[
					transforms.Resize((112, 112)),
					transforms.ToTensor(),
					transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
				]
			)

			# pass the face ROI through the trained liveness detector
			# model to determine if the face is "real" or "fake"
        face = transform(face)
        # output = model(face.unsqueeze(0).cuda())	
        output = model(face.unsqueeze(0))
        print(output)
        output = torch.where(output < 0.25, torch.zeros_like(output), torch.ones_like(output))
        if output.item() == 0:
            label = "{}".format("Live")
				# draw the label and bounding box on the frame
            cv2.putText(img, label, (startX, startY - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
        else:
            label = "{}".format("Spoof")
				# draw the label and bounding box on the frame
            cv2.putText(img, label, (startX, startY - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)

	# show the output frame and wait for a key press
    cv2.imshow("Frame", img)
    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
