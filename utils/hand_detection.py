from typing import Optional

import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import drawing_styles, drawing_utils
from mediapipe.python.solutions.hands import HAND_CONNECTIONS
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers import NormalizedLandmark

# from mediapipe.python.solutions import drawing_styles


class Hand_detector:
    __BASE_MODEL: str = "models/hand_landmarker.task"

    def __init__(
        self,
        number_of_hands: int = 2,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model: str = __BASE_MODEL,
    ) -> None:

        self.num_hands: int = number_of_hands
        self.min_hand_detection_confidence: float = (
            min_hand_detection_confidence
        )
        self.min_hand_presence_confidence: float = min_hand_presence_confidence
        self.min_tracking_confidence: float = min_tracking_confidence
        self.model = model

        # initializing mediapipe hand detector
        base_options = python.BaseOptions(self.model)
        options = vision.HandLandmarkerOptions(
            base_options,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.min_hand_detection_confidence,
            min_hand_presence_confidence=self.min_hand_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(options)

        # detection properties customization options
        self.draw_landmarks: bool = False
        self.draw_bounding_box: bool = False

        # ---
        self.__Processing_image: np.ndarray

    @staticmethod
    def __image_formate(image: np.ndarray):
        mp_img = mp.Image(
            mp.ImageFormat.SRGB, cv.cvtColor(image, cv.COLOR_BGR2RGB)
        )
        return mp_img

    def detect_hand_landmarks_in_image(
        self,
        image: np.ndarray,
        draw_landmarks: bool,
        handedness: list = ["Left", "Right"],
        only_landmarks: bool = False,
    ):
        self.__Processing_image = self.__image_formate(image)
        self.draw_landmarks = draw_landmarks
        landmarks_result = self.hand_detector.detect(self.__Processing_image)
        self.__hand_landmark_list: list = landmarks_result.hand_landmarks
        hands: list = landmarks_result.handedness
        detected_landmarks: list[dict] = []
        if only_landmarks:
            image = np.zeros([image.shape[0], image.shape[1], image.shape[3]])

        if self.draw_landmarks:
            if not hands:
                return image, detected_landmarks

            for idx in range(len(hands)):
                if hands[idx][0].category_name in handedness:

                    hand_land_marks = self.__hand_landmark_list[idx]
                    land_marks = landmark_pb2.NormalizedLandmarkList()
                    land_marks.landmark.extend(
                        [
                            NormalizedLandmark(
                                landmark.x, landmark.y, landmark.z
                            ).to_pb2()
                            for landmark in hand_land_marks
                        ]
                    )
                    lm_list = [
                        [landmark.x, landmark.y, landmark.z]
                        for landmark in hand_land_marks
                    ]
                    drawing_utils.draw_landmarks(
                        image,
                        land_marks,
                        connections=HAND_CONNECTIONS,
                        landmark_drawing_spec=drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=drawing_styles.get_default_hand_connections_style(),
                    )
                    detected_landmarks.append(
                        {
                            "handedness": hands[idx][0].category_name,
                            "lm_list": lm_list,
                        }
                    )
        return image, detected_landmarks
