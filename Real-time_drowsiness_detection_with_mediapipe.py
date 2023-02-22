# -*- coding: utf-8 -*-
"""
Author: A. N. M. Sajedul ALam
Email: anmsajedulalam@gmail.com
Goal: Drowsiness Detection Using Facial Landmarks
"""

import mediapipe as mp
import cv2 as cv
from scipy.spatial import distance as dis
import pyttsx3
import playsound

import threading
speech = pyttsx3.init()

def run_speech(speech,speech_message):
    playsound.playsound('storm.mp3', False)
    speech = pyttsx3.init()
    speech.say(speech_message)
    speech.runAndWait()
    pyttsx3.engine.Engine.stop(speech)





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
min_frame = 5
min_tolerance = 5.0


###STREAMLIT GUI CODE

import cv2
import streamlit as st

st.set_page_config(layout="wide")

hide_streamlit_style = """
            <style>
            button[title="View fullscreen"]{
            visibility: hidden;}
            div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }

            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}

            .css-15zrgzn {display: none}
            .css-eczf16 {display: none}
            .css-jn99sy {display: none}
            .css-v84420.e1tzin5v2 {text-align: center}
            
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 



st.image("https://bondstein.com/wp-content/uploads/2021/04/Bondstein-Logo.png", width=400)


st.markdown(hide_streamlit_style, unsafe_allow_html=True) 



st.title("Realtime Driver Tracking")

col1, col2 = st.columns([3, 1],gap="large")

with col1:
    FRAME_WINDOW = st.image([])

with col2:
    warning = st.image([])
    

    placeholder = st.empty()
    
st.subheader("Developed by Bondstein Technologies Limited")    
     

while True:

    
    result, image = capture.read()

    if result:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image= cv.flip(image, 1)
        #image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB) ### converting BGR to RGB
        outputs = face_model.process(image)

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
            else:
                if frame_count != 0:
                    frame_count =frame_count-1 
                

            if frame_count > min_frame:
                #### doing eye ratio analysis here
                #speech = pyttsx3.init()
                
                #message = 'Hey driver, it Seems you are sleeping, please wake up wake up wake up'
                #t =  threading.Thread(target=run_speech, args=(speech,message), daemon=True)
                t =  threading.Thread(target=playsound.playsound, args=('beep.mp3', False), daemon=True)
                #### here, i'm creating new instance if the thread is dead
                
                draw_landmarks(image, outputs, UPPER_LOWER_LIPS , COLOR_BLUE)
                draw_landmarks(image, outputs, LEFT_RIGHT_LIPS, COLOR_BLUE)
                #run_speech(speech, message)
                warning.image("redwarn.png",use_column_width="always")
                placeholder.header("KEEP YOUR EYES ON THE ROAD!!")
                #playsound.playsound('eyes.mp3', True)
                t.start()


            ratio_lips =  get_aspect_ratio(image, outputs, UPPER_LOWER_LIPS, LEFT_RIGHT_LIPS)
            if ratio_lips < 1.8:
                #### doing mouth ratio analysis here
                #speech = pyttsx3.init()

                #message = 'Hey driver, you are looking tired, please take rest take rest take rest'
                #p = threading.Thread(target=run_speech, args=(speech,message), daemon=True)
                p = threading.Thread(target=playsound.playsound, args=('beep.mp3', False), daemon=True)
                #### here, i'm creating new instance if the thread is dead
                
                #run_speech(speech, message)
                warning.image("yellowwarn.png",use_column_width="always")
                placeholder.header("YOU ARE TIRED!!")
                p.start()
            if ratio_lips > 1.8 and frame_count < min_frame:
                warning.image("green.png",use_column_width="always")
                
                
                placeholder.header("SAFE DRIVING")

        #cv.imshow("Sajid's Drowsiness Detector", image)
        image = cv2.resize(image, (1280,720))
    FRAME_WINDOW.image(image,use_column_width="always")
        

capture.release()
cv.destroyAllWindows()


