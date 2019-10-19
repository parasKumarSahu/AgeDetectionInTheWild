# import libraries
import cv2
import glob
import cvlib as cv
import numpy as np

for img in glob.glob("images/*.jpg"):
	im = cv2.imread(img)
	im_cpy = np.copy(im)
	faces, confidences = cv.detect_face(im)
	for face in faces:    
		X, Y, Z = im.shape
		(startX,startY) = face[0],face[1]
		(endX,endY) = face[2],face[3]    
		if 0 < startX and X > endX and 0 < startY and Y > endY:
			cv2.rectangle(im, (startX,startY), (endX,endY), (0,255,0), 2)
			startX2 = int(startX-(0.2*(endX-startX))) 
			if startX2 < 0:
				startX2 = 0
			startY2 = int(startY-(0.2*(endY-startY))) 
			if startY2 < 0:
				startY2 = 0
			endX2 = int(endX+(0.2*(endX-startX))) 
			if endX2 > X:
				endX2 = X
			endY2 = int(endY+(0.2*(endY-startY))) 
			if endY2 > Y:
				endY2 = Y
			cv2.rectangle(im, (startX,startY), (endX,endY), (0,255,0), 2) 
			cv2.rectangle(im, (startX2,startY2), (endX2,endY2), (0,0,255), 2) 
	img_name = img[7:]		
	cv2.imwrite("images/marked_face/"+img_name, im)
	print("images/marked_face/"+img_name)
	cv2.imwrite("images/cropped_face/"+img_name, im_cpy[startX2:endX2, startY2:endY2])
	print("images/cropped_face/"+img_name)
