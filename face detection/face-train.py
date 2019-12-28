import cv2

import os
import numpy as np
from PIL import Image
import pickle

base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir,"images")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()


current_ids = 0
labels_ids = {}
y_labels = []
x_train = []


for root , dirs , files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root,file)
			label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
			#print(label, path)
			if label in labels_ids:
				pass
			else:
				labels_ids[label]=current_ids
				current_ids += 1
			id_ = labels_ids[label]
			#print(labels_ids)		

			#y_labels.append(label)
			#x_train.append(path)
			pil_image = Image.open(path).convert("L")
			size =(550,550)
			final_image = pil_image.resize(size,Image.ANTIALIAS)
			image_array = np.array(final_image,"uint8")
			#print(image_array)
			faces = face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5)

			for (x,y,w,h) in faces:
				roi = image_array[y:y+h,x:x+w]
				x_train.append(roi)
				y_labels.append(id_)


#print(y_labels)
#print(x_train)				
with open("label.pickle","wb") as f:
	pickle.dump(labels_ids,f)


recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainer.yml")