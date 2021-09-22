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

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

pyautogui.FAILSAFE = False

IMAGE_WIDTH = None
IMAGE_HEIGHT = None
MAX_NUM_HANDS = 2


class ThreeDimensionalEnvironment:

    def __init__(self):
        self._landmarks = None
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111, projection='3d')

    @property
    def landmarks(self):
        return self._landmarks

    @property
    def fig(self):
        return self._fig

    def set_landmarks(self, landmarks):
        self._landmarks = landmarks
        self._ax.clear()
        self._ax.set_xlim3d(0, 1)
        self._ax.set_ylim3d(0, 1)
        self._ax.set_zlim3d(-1, 1)
        self._ax.set_xlabel('X Label')
        self._ax.set_ylabel('Y Label')
        self._ax.set_zlabel('Z Label')

        xs = []
        ys = []
        zs = []

        if type(landmarks) == list:
            for frame_landmark in landmarks:
                if hasattr(frame_landmark, 'landmark'):
                    for landmark in frame_landmark.landmark:
                        xs.append(landmark.x)
                        ys.append(landmark.y)
                        zs.append(landmark.z)

        else:

            for landmark in landmarks.landmark:
                xs.append(landmark.x)
                ys.append(landmark.y)
                zs.append(landmark.z)

        # alphas = np.repeat(np.linspace(1.0, 0.1, int(len(xs) / 21)), 21)
        # colors = np.append(np.repeat('blue', len(xs) - 21), np.repeat('red', 21))
        # self._ax.scatter(xs, ys, zs, zdir='z', c=colors, alpha=alphas, depthshade=False)
        self._ax.scatter(xs, ys, zs, zdir='z')
