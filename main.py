import math
import os
import time

import cv2
import mediapipe as mp
import numpy as np

import pyautogui
from three_dimensional_environment import ThreeDimensionalEnvironment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib import style
from typing import List

from hand import Hand

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

pyautogui.FAILSAFE = False

IMAGE_WIDTH = None
IMAGE_HEIGHT = None

MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE_HAND = 0.75
MIN_TRACKING_CONFIDENCE = 0.45


def main():
    prev_frame_time = 0

    hand = Hand()
    env = ThreeDimensionalEnvironment()
    plt.ion()

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE_HAND,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    ) as hands:
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                print('Ignore empty camera frame')
                break

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            new_landmarks = []

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    new_landmarks = hand_landmarks

                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            # print(type(new_landmarks))
            hand.update_cache(new_landmarks)
            env.set_landmarks(hand.landmarks_cache())

            cv2.putText(image, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            plt.pause(0.00000001)
            cv2.imshow('Camera', image)
            if cv2.waitKey(1) == ord('q'):
                break
    cap.release()
    plt.ioff()


if __name__ == "__main__":
    main()
