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
        self.__processed_image: np.ndarray = np.zeros([1, 1, 1])
        self.__processed_handlandmark: list
        self.__processed_handedness: list

        # initializing mediapipe hand detector
        base_options = python.BaseOptions(self.model, delegate="GPU")
        options = vision.HandLandmarkerOptions(
            base_options,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.min_hand_detection_confidence,
            min_hand_presence_confidence=self.min_hand_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(options)

        # detection properties customization options
        self.draw_landmarks: bool
        self.draw_landmarks: bool

    def __compare_image(self, image):
        if np.array_equal(self.__processed_image, image):
            hand_landmark_list, hands = (
                self.__processed_handlandmark,
                self.__processed_handedness,
            )
        else:
            hand_landmark_list, hands = self.__process_image(image)
            self.__processed_handlandmark, self.__processed_handedness = (
                hand_landmark_list,
                hands,
            )
        return hand_landmark_list, hands

    @staticmethod
    def __image_formate(image: np.ndarray):
        try:
            mp_img = mp.Image(
                mp.ImageFormat.SRGB, cv.cvtColor(image, cv.COLOR_BGR2RGB)
            )
            return mp_img
        except:
            return image

    def __process_image(self, image):

        mp_image = self.__image_formate(image)
        landmarks_result = self.hand_detector.detect(mp_image)
        hand_landmark_list: list = landmarks_result.hand_landmarks
        hands: list = landmarks_result.handedness
        return hand_landmark_list, hands

    def detect_hand_landmarks_in_image(
        self,
        image: np.ndarray,
        draw_landmarks: bool,
        handedness: list = ["Left", "Right"],
        only_landmarks: bool = False,
    ):

        hand_landmark_list, hands = self.__compare_image(image)
        self.draw_landmarks = draw_landmarks
        detected_landmarks: list[dict] = []
        if only_landmarks:
            image = np.zeros([image.shape[0], image.shape[1], image.shape[2]])

        if self.draw_landmarks:
            if not hands:
                self.__processed_image = image
                return image, detected_landmarks

            for idx in range(len(hands)):
                if hands[idx][0].category_name in handedness:

                    hand_land_marks = hand_landmark_list[idx]
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
                        landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
                        connection_drawing_spec=drawing_styles.get_default_hand_connections_style(),
                    )

                    detected_landmarks.append(
                        {
                            "handedness": hands[idx][0].category_name,
                            "lm_list": lm_list,
                        }
                    )
        self.__processed_image = image
        return image, detected_landmarks

    def draw_bounding_box(
        self,
        image: np.ndarray,
        offset_x: int = 20,
        offset_y: int = 20,
        const_bbox: bool = False,
        draw_bbox: bool = True,
        bbox_color: tuple[int, int, int] = (255, 0, 0),
    ):

        hand_landmark_list, _ = self.__compare_image(image)
        bbox_data = []
        img_height, img_width, _ = image.shape
        if hand_landmark_list:
            for hand_landmark in hand_landmark_list:
                # Scale landmark values to pixel values
                hand_landmark_x_values = [
                    int(landmarks.x * img_width) for landmarks in hand_landmark
                ]
                hand_landmark_y_values = [
                    int(landmarks.y * img_height) for landmarks in hand_landmark
                ]

                max_x, max_y = max(hand_landmark_x_values), max(
                    hand_landmark_y_values
                )
                min_x, min_y = min(hand_landmark_x_values), min(
                    hand_landmark_y_values
                )

                center = [
                    int(np.mean([max_x, min_x])),
                    int(np.mean([max_y, min_y])),
                ]
                # Determine the bounding box coordinates
                if not const_bbox:
                    pts_max = [max_x + offset_x, max_y + offset_y]
                    pts_min = [min_x - offset_x, min_y - offset_y]
                else:
                    pts_max = [center[0] + offset_x, center[1] + offset_y]
                    pts_min = [center[0] - offset_x, center[1] - offset_y]
                bbox_data.append([pts_max, pts_min, center])

                # Draw the rectangle
                if draw_bbox:
                    cv.rectangle(
                        image,
                        pts_min,
                        pts_max,
                        bbox_color,
                        thickness=4,
                        lineType=cv.LINE_8,
                    )
        self.__processed_image = image
        return image, bbox_data

    def get_cropped_image(self):
        # TODO:
        pass
