import cv2
import pickle

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 250)
cap.set(cv2.CAP_PROP_FPS, 25)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

labels = {'person_name': 1}
with open('labels.pickle', 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

while True:
    # capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=2,
        minNeighbors=5
    )
    for (x, y, w, h) in faces:
        # print(x, y, w, h)
        # roi stands for region of interest
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # recognize?
        id_, conf = recognizer.predict(roi_gray)
        if 45 <= conf <= 85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_PLAIN
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(
                frame, name,
                (x, y), font,
                1, color,
                stroke, cv2.LINE_AA)
        # save img
        img_item = 'my_image.png'
        cv2.imwrite(img_item, roi_gray)

        # draw rectangle
        # BGR color
        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(
            frame,
            (x, y),
            (end_cord_x, end_cord_y),
            color,
            stroke
        )
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color,
                          (ex, ey),
                          (ex+ew, ey+eh),
                          (0, 255, 0), 2)
    # Display the resulting time
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
