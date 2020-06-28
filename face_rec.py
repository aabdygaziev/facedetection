import os
import numpy as np
import cv2
from PIL import Image
import pickle

base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, 'images')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 250)
cap.set(cv2.CAP_PROP_FPS, 25)

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("JPG"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            # print(label, path)
            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            # print(label_ids)
            # y_labels.append(label) # some number
            # x_train.append(path) # verify this image, turn into a NUMPY arrray, GRAY
            pil_image = Image.open(path).convert("L")  # grayscale
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8")
            # print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = image_array[y:y + h, x:x + w]
                x_train.append(roi)
                y_labels.append(id_)

print(y_labels)
print(label_ids)

with open('labels.pickle','wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save('trainer.yml')
