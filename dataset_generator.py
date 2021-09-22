import math
import time

import cv2
import mediapipe as mp
import numpy as np

import pyautogui

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib import style

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

IMAGE_WIDTH = None
IMAGE_HEIGHT = None
MAX_NUM_HANDS = 2

hand_hold = {
    'fist': 'f',
    'open': 'o',
    'pointing': 'p',
}


def add_data():
    pass


cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # line_coords = cursor.hand_line(image)
                # cv2.line(image, line_coords['thumb_cmc'], line_coords['pinky_mcp'], (0, 0, 255), 2)
                # cv2.line(image, line_coords['line'][0], line_coords['line'][1], (0, 0, 255), 2)
                # cv2.line(image, line_coords['perp_line'][0], line_coords['perp_line'][1], (0, 255, 255), 2)

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
