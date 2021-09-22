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

from three_dimensional_environment import ThreeDimensionalEnvironment

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

pyautogui.FAILSAFE = False

IMAGE_WIDTH = None
IMAGE_HEIGHT = None
MAX_NUM_HANDS = 2


class Hand:

    def __init__(self):
        self._landmarks = None
        self._capture_motion = False
        self._fingers_opened = [False, False, False, False, False]

        self._landmarks_cache = []
        self._fingers_opened_cache = []

    def set_landmarks(self, landmarks):
        self._landmarks = landmarks
        # self.palm_direction()
        # if self._capture_motion:
        #   self.hand_hold_recognition()

    def update_cache(self, new_landmarks=None, cached_ms=3):
        if len(self._landmarks_cache) >= cached_ms:
            del self._landmarks_cache[0]
        self._landmarks_cache.append(new_landmarks)
        assert len(self._landmarks_cache) <= cached_ms

    def plot_cache(self):
        plt.ion()
        env = ThreeDimensionalEnvironment()
        env.set_landmarks(self._landmarks_cache)
        plt.ioff()

    def palm_measure(self, flat=False):
        thumb_cmc = self._landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
        index_finger_mcp = self._landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        pinky_mcp = self._landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

        if flat:
            d_thumb_index = self.distance_two_dimensional(
                thumb_cmc.x, thumb_cmc.y,
                index_finger_mcp.x, index_finger_mcp.y,
            )
            d_index_pinky = self.distance_two_dimensional(
                index_finger_mcp.x, index_finger_mcp.y,
                pinky_mcp.x, pinky_mcp.y,
            )
            d_thumb_pinky = self.distance_two_dimensional(
                thumb_cmc.x, thumb_cmc.y,
                pinky_mcp.x, pinky_mcp.y,
            )
        else:
            d_thumb_index = self.distance_three_dimensional(
                thumb_cmc.x, thumb_cmc.y, thumb_cmc.z,
                index_finger_mcp.x, index_finger_mcp.y, index_finger_mcp.z
            )
            d_index_pinky = self.distance_three_dimensional(
                index_finger_mcp.x, index_finger_mcp.y, index_finger_mcp.z,
                pinky_mcp.x, pinky_mcp.y, pinky_mcp.z
            )
            d_thumb_pinky = self.distance_three_dimensional(
                thumb_cmc.x, thumb_cmc.y, thumb_cmc.z,
                pinky_mcp.x, pinky_mcp.y, pinky_mcp.z
            )
        return d_thumb_index, d_index_pinky, d_thumb_pinky

    def avg_hand_size(self, flat=False):
        x, y, z = self.palm_measure(flat=flat)
        return (x + y + z) / 3

    def palm_direction(self, tolerance=0.65):
        x, y, z = self.palm_measure(flat=True)
        avg_hand_size = self.avg_hand_size(flat=True)
        capture_motion = True
        for value in [x, y, z]:
            if value < avg_hand_size * tolerance:
                capture_motion = False
        self._capture_motion = capture_motion

    @staticmethod
    def distance_three_dimensional(x1, y1, z1, x2, y2, z2):
        return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2) + math.pow(z1 - z2, 2))

    @staticmethod
    def distance_two_dimensional(x1, y1, x2, y2):
        return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

    def hand_hold_recognition(self, tolerance=0.95):
        fingers = {
            'thumb': {
                'detection_points': [
                    (mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.THUMB_TIP),
                    (mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.THUMB_TIP),
                ],
            },
            'index': {
                'detection_points': (mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.INDEX_FINGER_TIP),
            },
            'middle': {
                'detection_points': (mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP),
            },
            'ring': {
                'detection_points': (mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_TIP),
            },
            'pinky': {
                'detection_points': (mp_hands.HandLandmark.PINKY_MCP, mp_hands.HandLandmark.PINKY_TIP),
            },
        }

        avg_hand_size = self.avg_hand_size()
        position = []
        for name, finger in fingers.items():
            if type(finger['detection_points']) == list:
                d = 0
                for points in finger['detection_points']:
                    p0 = self._landmarks.landmark[points[0]]
                    p1 = self._landmarks.landmark[points[1]]
                    d += self.distance_three_dimensional(p0.x, p0.y, p0.z, p1.x, p1.y, p1.z)
                d = d / len(finger['detection_points'])

            else:
                p0 = self._landmarks.landmark[finger['detection_points'][0]]
                p1 = self._landmarks.landmark[finger['detection_points'][1]]
                d = self.distance_three_dimensional(p0.x, p0.y, p0.z, p1.x, p1.y, p1.z)

            if d > avg_hand_size * tolerance:
                position.append(True)
            else:
                position.append(False)
        self._fingers_opened = position

    def fingers(self) -> List[bool]:
        return self._fingers_opened

    def capture(self):
        return self._capture_motion

    def landmarks_cache(self):
        return self._landmarks_cache
