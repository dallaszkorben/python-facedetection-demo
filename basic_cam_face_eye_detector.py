#!/usr/bin/python
# This script will detect faces via your webcam.
# Tested with OpenCV3

import cv2

cap = cv2.VideoCapture(0)

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarclassifiers/haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("haarclassifiers/haarcascade_eye_tree_eyeglasses.xml")

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	if not ret:
		print("No camera found...")
		break
		
	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
		#flags = cv2.CV_HAAR_SCALE_IMAGE
	)

	eyes = eyeCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=20,
		minSize=(10, 10)
		#flags = cv2.CV_HAAR_SCALE_IMAGE
	)

	print("Found {0} faces! Found {1} eyes".format(len(faces), len(eyes)))

	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)		

	# Draw a circle around the eye
	for (x, y, w, h) in eyes:
		cv2.circle(frame, ( int( x + w / 2), int( y + w / 2) ), int( w / 2 ), (0, 0, 255), 2)

	# Display the resulting frame
	#cv2.imshow('frame', frame)	
	cv2.imshow('frame', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
