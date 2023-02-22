import cv2
import face_recognition
from simple_facerec import SimpleFacerec


sfr = SimpleFacerec()
sfr.load_encoding_images('images/')

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    face_location, face_name =sfr.detect_known_faces(frame)

    for face_loc, name in zip(face_location, face_name):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()