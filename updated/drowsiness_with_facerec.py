# -*- coding: utf-8 -*-
"""
Author: Bondstein ML Team
Email: 
Goal: 
"""

import mediapipe as mp
import cv2 as cv
from scipy.spatial import distance as dis
from simple_facerec import SimpleFacerec


sfr = SimpleFacerec()
sfr.load_encoding_images('images/')


import threading

import pyttsx3


def run_speech(speech, speech_message):
    speech.say(speech_message)
    speech.runAndWait()


def draw_landmarks(image, outputs, land_mark, color): ## This function draws facial landmarks
    height, width =image.shape[:2]

    for face in land_mark:
        point = outputs.multi_face_landmarks[0].landmark[face]

        point_scale = ((int)(point.x * width), (int)(point.y*height))

        cv.circle(image, point_scale, 2, color, 1)

def euclidean_distance(image, top, bottom): ## This function calculates euclidean distance of the specific points or landmarks
    height, width = image.shape[0:2]

    point1 = int(top.x * width), int(top.y * height)
    point2 = int(bottom.x * width), int(bottom.y * height)

    distance = dis.euclidean(point1, point2)
    return distance


def get_aspect_ratio(image, outputs, top_bottom, left_right): ## This function calculates aspect ratio
    landmark = outputs.multi_face_landmarks[0]

    top = landmark.landmark[top_bottom[0]]
    bottom = landmark.landmark[top_bottom[1]]

    top_bottom_dis = euclidean_distance(image, top, bottom)

    left = landmark.landmark[left_right[0]]
    right = landmark.landmark[left_right[1]]

    left_right_dis = euclidean_distance(image, left, right)

    aspect_ratio = left_right_dis/ top_bottom_dis

    return aspect_ratio


face_mesh = mp.solutions.face_mesh
draw_utils = mp.solutions.drawing_utils
landmark_style = draw_utils.DrawingSpec((0,255,0), thickness=1, circle_radius=1)
connection_style = draw_utils.DrawingSpec((0,0,255), thickness=1, circle_radius=1)


STATIC_IMAGE = False
MAX_NO_FACES = 2
DETECTION_CONFIDENCE = 0.6
TRACKING_CONFIDENCE = 0.5

COLOR_RED = (0,0,255)
COLOR_BLUE = (255,0,0)
COLOR_GREEN = (0,255,0)


### For tracking lips, using these specific landmarks.
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
       185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]

### For tracking right eye, using these specific landmarks.
RIGHT_EYE = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
### For tracking left eye, using these specific landmarks.
LEFT_EYE = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]

### For tracking left eye top bottom position, using these specific landmarks.
LEFT_EYE_TOP_BOTTOM = [386, 374]
### For tracking left eye left right position, using these specific landmarks.
LEFT_EYE_LEFT_RIGHT = [263, 362]

### For tracking right eye top bottom position, using these specific landmarks.
RIGHT_EYE_TOP_BOTTOM = [159, 145]
### For tracking right eye left right position, using these specific landmarks.
RIGHT_EYE_LEFT_RIGHT = [133, 33]

UPPER_LOWER_LIPS = [13, 14]
LEFT_RIGHT_LIPS = [78, 308]

### For tracking face, using these specific landmarks.
FACE=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
       377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

### generating the face model here
face_model = face_mesh.FaceMesh(static_image_mode=STATIC_IMAGE,
                                max_num_faces= MAX_NO_FACES,
                                min_detection_confidence=DETECTION_CONFIDENCE,
                                min_tracking_confidence=TRACKING_CONFIDENCE)

### capturing input starts from here
capture = cv.VideoCapture(0) ### taking input from webcam

### frame counts
frame_count = 0
min_frame = 6
min_tolerance = 5.0

### using pyttsx3 for speech text to speech output
speech = pyttsx3.init()

while True:
    result, image = capture.read()

    if result:
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB) ### converting BGR to RGB

        face_location, face_name =sfr.detect_known_faces(image)

        for face_loc, name in zip(face_location, face_name):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            cv.putText(image, name, (x1, y1-10), cv.FONT_HERSHEY_DUPLEX, 1, (0,0,200), 2)
            cv.rectangle(image, (x1, y1), (x2, y2), (0, 0, 200), 4)

        outputs = face_model.process(image_rgb)

        if outputs.multi_face_landmarks:

            draw_landmarks(image, outputs, FACE, COLOR_GREEN)


            draw_landmarks(image, outputs, LEFT_EYE_TOP_BOTTOM, COLOR_RED)
            draw_landmarks(image, outputs, LEFT_EYE_LEFT_RIGHT, COLOR_RED)

            ratio_left =  get_aspect_ratio(image, outputs, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT)


            draw_landmarks(image, outputs, RIGHT_EYE_TOP_BOTTOM, COLOR_RED)
            draw_landmarks(image, outputs, RIGHT_EYE_LEFT_RIGHT, COLOR_RED)

            ratio_right =  get_aspect_ratio(image, outputs, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT)

            ratio = (ratio_left + ratio_right)/2.0

            if ratio > min_tolerance:
                frame_count +=1
                cv.putText(image, 'Sleeping', (0+20,0+25), cv.FONT_HERSHEY_DUPLEX, 1, (0,0,200), 2)
            else:
                frame_count = 0

            if frame_count > min_frame:
                #### doing eye ratio analysis here

                message = 'Hey driver, it Seems you are sleeping, please wake up wake up wake up'
                t = threading.Thread(target=run_speech, args=(speech, message))
                #### here, i'm creating new instance if the thread is dead
                t.start()



            draw_landmarks(image, outputs, UPPER_LOWER_LIPS , COLOR_BLUE)
            draw_landmarks(image, outputs, LEFT_RIGHT_LIPS, COLOR_BLUE)


            ratio_lips =  get_aspect_ratio(image, outputs, UPPER_LOWER_LIPS, LEFT_RIGHT_LIPS)
            if ratio_lips < 1.8:
                #### doing mouth ratio analysis here

                cv.putText(image, 'Tired', (0+20,0+25), cv.FONT_HERSHEY_DUPLEX, 1, (0,0,200), 2)
                message = 'Hey driver, you are looking tired, please take rest take rest take rest'
                p = threading.Thread(target=run_speech, args=(speech, message))
                #### here, i'm creating new instance if the thread is dead
                p.start()

        cv.imshow("Driver Identification and Drowsiness Detection", image)
        if cv.waitKey(1) & 255 == 27:
            break


capture.release()
cv.destroyAllWindows()