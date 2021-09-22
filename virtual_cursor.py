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
from typing import List

from hand import Hand

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

pyautogui.FAILSAFE = False

IMAGE_WIDTH = None
IMAGE_HEIGHT = None
MAX_NUM_HANDS = 2


class Cursor:
    _counter = 0

    def __init__(self, image_width=640, image_height=480):
        self._num = Cursor._counter
        Cursor._counter += 1

        self._landmarks = None
        self._x = None
        self._y = None
        self._click = False

        self._screen_width = pyautogui.size()[0]
        self._screen_height = pyautogui.size()[1]
        self._image_width = image_width
        self._image_height = image_height

        self._cursor_landmark = mp_hands.HandLandmark.INDEX_FINGER_TIP
        self._detecting_area_landmark = mp_hands.HandLandmark.INDEX_FINGER_PIP

    @staticmethod
    def distance_from_point(x_p, y_p, x_c, y_c):
        return math.sqrt(math.pow(x_p - x_c, 2) + math.pow(y_p - y_c, 2))

    def landmarks(self, landmarks, image):
        self._landmarks = landmarks
        self.set_cursor_position(landmarks)
        self.relative_move_cursor(image)

    def set_cursor_position(self, landmarks):
        self._x = landmarks.landmark[self._cursor_landmark].x
        self._y = landmarks.landmark[self._cursor_landmark].y

    def hand_line(self, image):
        thumb_cmc = self._landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
        pinky_mcp = self._landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
        index_finger_mcp = self._landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

        image_width = image.shape[1]
        image_height = image.shape[0]

        thumb_cmc_pos = (int(thumb_cmc.x * image_width), int(thumb_cmc.y * image_height))
        pinky_mcp_pos = (int(pinky_mcp.x * image_width), int(pinky_mcp.y * image_height))
        index_finger_mcp_pos = (int(index_finger_mcp.x * image_width), int(index_finger_mcp.y * image_height))

        slope = 0
        if (thumb_cmc_pos[0] - pinky_mcp_pos[0]) != 0:
            slope = (thumb_cmc_pos[1] - pinky_mcp_pos[1]) / (thumb_cmc_pos[0] - pinky_mcp_pos[0])
        # c = y - mx
        intercept = thumb_cmc_pos[1] - slope * thumb_cmc_pos[0]

        perp_slope = 0
        if slope != 0:
            perp_slope = -1 / slope
        perp_intercept = index_finger_mcp_pos[1] - perp_slope * index_finger_mcp_pos[0]

        return {
            'thumb_cmc': thumb_cmc_pos,
            'pinky_mcp': pinky_mcp_pos,
            'index_finger_mcp': index_finger_mcp_pos,
            'line': ((0, int(intercept)), (image_width, int(image_width * slope + intercept))),
            'perp_line': ((0, int(perp_intercept)), (image_width, int(image_width * perp_slope + perp_intercept))),
        }

    def drag(self, d, distance_tolerance, active=True, verbose=False):
        if d <= distance_tolerance:
            if not self._click:
                if verbose:
                    print('down')
                if active:
                    pyautogui.mouseDown(button='left')
                self._click = True
        else:
            if self._click:
                if verbose:
                    print('up')
                if active:
                    pyautogui.mouseUp(button='left')
                self._click = False

    def click(self, d, distance_tolerance):
        self._click = False
        if d <= distance_tolerance:
            self._click = True
            print('click')
            # pyautogui.click()

    def detect_click(self, avg_hand_size, distance_tolerance=0.05, draggable=False):
        x_p = self._landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
        y_p = self._landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
        x_c = self._landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
        y_c = self._landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        d = self.distance_from_point(x_p, y_p, x_c, y_c)
        # print(d * (avg_hand_size / self._image_width))
        if draggable:
            self.drag(d, distance_tolerance)
        else:
            self.click(d, distance_tolerance)

    def relative_move_cursor(self, image, speed=0.3):

        detecting_area_landmark = self._landmarks.landmark[self._detecting_area_landmark]
        center_x = int(detecting_area_landmark.x * self._image_width)
        center_y = int(detecting_area_landmark.y * self._image_height)

        index_finger_mcp = self._landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        pinky_mcp = self._landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
        thumb_cmc = self._landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]

        fist_size = self.distance_from_point(
            pinky_mcp.x * self._image_width, pinky_mcp.y * self._image_height,
            index_finger_mcp.x * self._image_width, index_finger_mcp.y * self._image_height,
        )
        palm_size = self.distance_from_point(
            thumb_cmc.x * self._image_width, thumb_cmc.y * self._image_height,
            index_finger_mcp.x * self._image_width, index_finger_mcp.y * self._image_height,
        )
        avg_hand_size = (palm_size + fist_size) / 2

        d = self.distance_from_point(
            self._x * self._image_width, self._y * self._image_height,
            center_x, center_y
        )
        station_radius = int(avg_hand_size * 0.5)
        moving_radius = int(avg_hand_size * .9)
        if True:
            cv2.circle(image, (center_x, center_y), station_radius, (0, 0, 255), 1)
            cv2.circle(image, (center_x, center_y), moving_radius, (0, 255, 0), 1)
            # cv2.putText(image, str(avg_hand_size), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

        if d < station_radius:
            # print('click')
            self.detect_click(avg_hand_size)
        else:
            self._click = False

        if moving_radius > d >= station_radius:
            # print('moving')
            pyautogui.moveRel(
                (self._x * self._image_width - center_x) * (d - station_radius) * speed,
                (self._y * self._image_height - center_y) * (d - station_radius) * speed,
            )

    @property
    def cursor_position(self):
        return {'x': self._x, 'y': self._y, 'num': self._num}

    @property
    def is_click(self):
        return self._click






cursor = Cursor()
hand = Hand()
# plt.ion()
# plot = ThreeDimensionalEnvironment()

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.45
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

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand.set_landmarks(hand_landmarks)

                # cache = hand.landmarks_cache()

                # plot.set_landmarks(cache)

                # line_coords = cursor.hand_line(image)
                # cv2.line(image, line_coords['thumb_cmc'], line_coords['pinky_mcp'], (0, 0, 255), 2)
                # cv2.line(image, line_coords['line'][0], line_coords['line'][1], (0, 0, 255), 2)
                # cv2.line(image, line_coords['perp_line'][0], line_coords['perp_line'][1], (0, 255, 255), 2)

                # capture = hand.capture()
                # capture = False
                # if capture:
                #
                #
                #     cursor.landmarks(hand_landmarks, image)
                #
                #     capture_color = (0, 255, 0) if capture else (0, 0, 255)
                #     cv2.circle(image, (300, 300), 10, capture_color, -1)
                #
                #     fingers = hand.fingers()
                #     for idx, val in enumerate(fingers):
                #         color = (0, 255, 0) if val else (0, 0, 255)
                #         cv2.circle(image, ((idx + 1) * 25, 25), 10, color, -1)

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            # if cursor.is_click:
            #     cv2.circle(image, (25, 25), 10, (0, 0, 255), -1)

        # plt.pause(0.000000001)
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
