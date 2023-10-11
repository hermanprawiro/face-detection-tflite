# -*- coding: utf-8 -*-
# Copyright © 2021 Patrick Levin
# SPDX-Identifier: MIT
import os
import numpy as np
import tensorflow as tf
from PIL.Image import Image
from typing import List, Optional, Sequence, Tuple, Union
from . import ModelDataError
from .render import Annotation
from .render import Color, landmarks_to_render_data
from .transform import SizeMode, bbox_to_roi, image_to_tensor, sigmoid
from .transform import project_landmarks
from .types import Detection, Landmark, Rect
from .face_detection import FaceIndex
from .iris_landmark import LEFT_EYE_TO_FACE_LANDMARK_INDEX, RIGHT_EYE_TO_FACE_LANDMARK_INDEX
"""Model for face landmark detection with Attention Mesh.

Ported from Google® MediaPipe (https://google.github.io/mediapipe/).

Model card:

    https://drive.google.com/file/d/1tV7EJb3XgMS7FwOErTgLU1ZocYyNmwlf/preview

Reference:

    Attention Mesh: High-fidelity Face Mesh Prediction 
    in Real-time, CVPR Workshop on Computer
    Vision for Augmented and Virtual Reality, Seattle,
    WA, USA, 2020
"""

MODEL_NAME = 'face_landmark_with_attention.tflite'
NUM_DIMS = 3                    # x, y, z
NUM_LANDMARKS = 468             # number of points in the face mesh
ROI_SCALE = (1.5, 1.5)          # Scaling of the face detection ROI
DETECTION_THRESHOLD = 0.5       # minimum score for detected faces

# face landmark connections
# (from face_landmarks_to_render_data_calculator.cc)
FACE_LANDMARK_CONNECTIONS = [
    # Lips.
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),
    (314, 405), (405, 321), (321, 375), (375, 291), (61, 185), (185, 40),
    (40, 39), (39, 37), (37, 0), (0, 267), (267, 269),
    (269, 270), (270, 409), (409, 291), (78, 95), (95, 88), (88, 178),
    (178, 87), (87, 14), (14, 317), (317, 402), (402, 318), (318, 324),
    (324, 308), (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312),
    (312, 311), (311, 310), (310, 415), (415, 308),
    # Left eye.
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
    (154, 155), (155, 133), (33, 246), (246, 161), (161, 160), (160, 159),
    (159, 158), (158, 157), (157, 173), (173, 133),
    # Left eyebrow.
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66),
    (66, 107),
    # Right eye.
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381),
    (381, 382), (382, 362), (263, 466), (466, 388), (388, 387), (387, 386),
    (386, 385), (385, 384), (384, 398), (398, 362),
    # Right eyebrow.
    (276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334),
    (334, 296), (296, 336),
    # Face oval.
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
]
MAX_FACE_LANDMARK = len(FACE_LANDMARK_CONNECTIONS)

LIPS_TO_FACE_LANDMARK_INDEX = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 
    291, 185, 40, 39, 37, 0, 267, 269, 270, 409, 
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 191, 80, 81, 82, 13, 312, 311, 310, 415, 
    76, 77, 90, 180, 85, 16, 315, 404, 320, 307,
    306, 184, 74, 73, 72, 11, 302, 303, 304, 408,
    62, 96, 89, 179, 86, 15, 316, 403, 319, 325,
    292, 183, 42, 41, 38, 12, 268, 271, 272, 407,
]


class FaceLandmarkAttention:
    """Face Landmark detection model as used by Google MediaPipe.

    This model detects facial landmarks from a face image.

    The model is callable and accepts a PIL image instance, image file name,
    and Numpy array of shape (height, width, channels) as input. There is no
    size restriction, but smaller images are processed faster.

    The output of the model is a list of 468 face landmarks in normalized
    coordinates (e.g. in the range [0, 1]).

    The preferred usage is to pass an ROI returned by a call to the
    `FaceDetection` model along with the image.

    Raises:
        ModelDataError: `model_path` points to an incompatible model
    """
    def __init__(
        self,
        model_path: Optional[str] = None
    ) -> None:
        if model_path is None:
            my_path = os.path.abspath(__file__)
            model_path = os.path.join(os.path.dirname(my_path), 'data')
        self.model_path = os.path.join(model_path, MODEL_NAME)
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.input_shape = self.interpreter.get_input_details()[0]['shape']
        
        self.data_index = self.interpreter.get_output_details()[3]['index']
        self.face_index = self.interpreter.get_output_details()[2]['index']
        self.data_left_eye_index = self.interpreter.get_output_details()[0]['index']
        self.data_right_eye_index = self.interpreter.get_output_details()[1]['index']
        self.data_lips_index = self.interpreter.get_output_details()[5]['index']
        self.data_left_iris_index = self.interpreter.get_output_details()[4]['index']
        self.data_right_iris_index = self.interpreter.get_output_details()[6]['index']

        data_shape = self.interpreter.get_output_details()[3]['shape']
        num_exected_elements = NUM_DIMS * NUM_LANDMARKS
        if data_shape[-1] < num_exected_elements:
            raise ModelDataError(f'incompatible model: {data_shape} < '
                                 f'{num_exected_elements}')
        self.interpreter.allocate_tensors()

    def __call__(
        self,
        image: Union[Image, np.ndarray, str],
        roi: Optional[Rect] = None
    ) -> List[Landmark]:
        """Run inference and return detections from a given image

        Args:
            image (Image|ndarray|str): Numpy array of shape
                `(height, width, 3)` or PIL Image instance or path to image.

            roi (Rect|None): Region within the image that contains a face.

        Returns:
            (list) List of face landmarks in nromalised coordinates relative to
            the input image, i.e. values ranging from [0, 1].
        """
        height, width = self.input_shape[1:3]
        image_data = image_to_tensor(
            image,
            roi,
            output_size=(width, height),
            keep_aspect_ratio=False,
            output_range=(0., 1.))
        input_data = image_data.tensor_data[np.newaxis]
        self.interpreter.set_tensor(self.input_index, input_data)
        self.interpreter.invoke()
        raw_data = self.interpreter.get_tensor(self.data_index)
        raw_face = self.interpreter.get_tensor(self.face_index)
        raw_left_eye = self.interpreter.get_tensor(self.data_left_eye_index)
        raw_right_eye = self.interpreter.get_tensor(self.data_right_eye_index)
        raw_lips = self.interpreter.get_tensor(self.data_lips_index)
        # second tensor contains confidence score for a face detection
        face_flag = sigmoid(raw_face).flatten()[-1]
        # no data if no face was detected
        if face_flag <= DETECTION_THRESHOLD:
            return []
        # combine all landmarks
        combine_raw_results(raw_data, raw_left_eye, raw_right_eye, raw_lips)
        # extract and normalise landmark data
        height, width = self.input_shape[1:3]
        return project_landmarks(raw_data,
                                 tensor_size=(width, height),
                                 image_size=image_data.original_size,
                                 padding=image_data.padding,
                                 roi=roi)


def pad_landmarks_2d_to_3d(
    data: np.ndarray,
) -> np.ndarray:
    """Pad 2D landmarks with zeros to convert them to 3D landmarks.

    Args:
        data (ndarray): Numpy array of shape `(num_points, 2)`.
    Returns:
        (ndarray) Numpy array of shape `(num_points, 3)`.
    """
    # normalize input type
    points = data.reshape(-1, 2)
    # add z coordinate
    points = np.concatenate([points, np.zeros((points.shape[0], 1))], axis=1)
    return points

def combine_raw_results(
    face_landmarks: np.ndarray,
    left_eye_landmarks: np.ndarray,
    right_eye_landmarks: np.ndarray,
    lips_landmarks: np.ndarray,
)-> np.ndarray:
    """Combine raw results from the model into a single array.

    Args:
        face_landmarks (ndarray): Numpy array of shape `(num_points, 3)`.
        left_eye_landmarks (ndarray): Numpy array of shape `(num_points, 2)`.
        right_eye_landmarks (ndarray): Numpy array of shape `(num_points, 2)`.
        lips_landmarks (ndarray): Numpy array of shape `(num_points, 2)`.
    Returns:
        (ndarray) Numpy array of shape `(num_points, 3)`.
    """
    # normalize input type
    face_landmarks = face_landmarks.reshape(-1, 3)
    left_eye_landmarks = pad_landmarks_2d_to_3d(left_eye_landmarks)
    right_eye_landmarks = pad_landmarks_2d_to_3d(right_eye_landmarks)
    lips_landmarks = pad_landmarks_2d_to_3d(lips_landmarks)

    # combine results
    for n, point in enumerate(left_eye_landmarks):
        index = LEFT_EYE_TO_FACE_LANDMARK_INDEX[n]
        face_landmarks[index] = point
    for n, point in enumerate(right_eye_landmarks):
        index = RIGHT_EYE_TO_FACE_LANDMARK_INDEX[n]
        face_landmarks[index] = point
    for n, point in enumerate(lips_landmarks):
        index = LIPS_TO_FACE_LANDMARK_INDEX[n]
        face_landmarks[index] = point
    return face_landmarks

    