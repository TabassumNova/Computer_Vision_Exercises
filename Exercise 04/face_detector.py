import cv2
import matplotlib.pyplot as plt
from mtcnn import MTCNN
import numpy as np


# The FaceDetector class provides methods for detection, tracking, and alignment of faces.
class FaceDetector:

    # Prepare the face detector; specify all parameters used for detection, tracking, and alignment.
    def __init__(self, tm_window_size=20, tm_threshold=0.7, aligned_image_size=224):
        # Prepare face alignment.
        self.detector = MTCNN()

        # Reference (initial face detection) for template matching.
        self.reference = None

        # Size of face image after landmark-based alignment.
        self.aligned_image_size = aligned_image_size
        self.tm_window_size = tm_window_size
        # self.w = 20
        # self.h = 20

	# ToDo: Specify all parameters for template matching.
        self.tm_threshold = tm_threshold
        self.template = None
        self.match_percentage = 0
        self.w, self.h = 0, 0

    # ToDo: Track a face in a new image using template matching.
    def track_face(self, image):
        while not self.reference:
            self.detector = MTCNN()
            self.reference = self.detect_face(image)
        if self.reference or (self.match_percentage >= self.tm_threshold):
            rect = self.reference["rect"]
            self.w, self.h = rect[2], rect[3]
            self.template = self.crop_face(self.reference["image"], rect)
        elif (self.match_percentage < self.tm_threshold):
            self.detector = MTCNN()
            self.reference = self.detect_face(image)
        output_image = {}
        output_image["image"] = image

        res = cv2.matchTemplate(image, self.template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        self.match_percentage = max_val
        output_image["response"] = max_val
        top_left = max_loc
        bottom_right = (top_left[0] + self.w, top_left[1] + self.h)
        output_image["rect"] = [top_left[0], top_left[1], self.w, self.h]
        output_aligned = self.align_face(image, output_image["rect"])
        output_image["aligned"] = output_aligned
        cv2.rectangle(image, top_left, bottom_right, 255, 2)

        # reference = self.reference
        # rect = reference["rect"]
        # # rect = [x+self.tm_window_size for x in reference["rect"]]
        # template = self.crop_face(reference["image"], rect)
        # self.w, self.h = template.shape[1], template.shape[0]
        # reference["image"] = cv2.matchTemplate(reference["image"], template, cv2.TM_CCOEFF_NORMED)
        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(reference["image"])
        # top_left = max_loc
        # bottom_right = (top_left[0] + self.w, top_left[1] + self.h)
        # reference["image"] = cv2.rectangle(reference["image"], top_left, bottom_right, 255, 2)
        # self.reference = reference
        # ### another method
        # loc = np.where(reference["image"] >= self.tm_threshold)
        # for pt in zip(*loc[::-1]):
        #     cv2.rectangle(reference["image"], pt, (pt[0] + self.w, pt[1] + self.h), (0, 0, 255), 2)
        # self.reference = reference
        return output_image

    # Face detection in a new image.
    def detect_face(self, image):
        # Retrieve all detectable faces in the given image.
        detections = self.detector.detect_faces(image)
        if not detections:
            self.reference = None
            return None

        # Select face with largest bounding box.
        largest_detection = np.argmax([d["box"][2] * d["box"][3] for d in detections])
        face_rect = detections[largest_detection]["box"]
        detection_probability = detections[largest_detection]["confidence"]
        # Align the detected face.
        aligned = self.align_face(image, face_rect)
        return {"rect": face_rect, "image": image, "aligned": aligned, "response": detection_probability}

    # Face alignment to predefined size.
    def align_face(self, image, face_rect):
        return cv2.resize(self.crop_face(image, face_rect), dsize=(self.aligned_image_size, self.aligned_image_size))

    # Crop face according to detected bounding box.
    def crop_face(self, image, face_rect):
        top = max(face_rect[1], 0)
        left = max(face_rect[0], 0)
        bottom = min(face_rect[1] + face_rect[3] - 1, image.shape[0] - 1)
        right = min(face_rect[0] + face_rect[2] - 1, image.shape[1] - 1)
        return image[top:bottom, left:right, :]

