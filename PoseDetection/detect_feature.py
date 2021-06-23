import mediapipe as mp
import cv2
from insukfile import drawing_utils as drawing

class FeatureDetector:
    def __init__(self):
        # self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing = drawing
        self.mp_holistic = mp.solutions.holistic
    def detectFeaturePoints(self, frame, image_start:list, image_end:list):
        mp_drawing = self.mp_drawing
        mp_holistic = self.mp_holistic

        for i in range(len(image_start)):
            start = tuple(image_start[i])
            end = tuple(image_end[i])
            # image -> human 영역만 crop 된 이미지
            image = frame[start[1]:end[1], start[0]:end[0]]  # 이미지[Y좌표, X좌표]
            # Convert the BGR image to RGB before processing.
            with mp_holistic.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.5) as holistic:
                results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                annotated_image = image.copy()  # 찾은 사람 이미지
                mp_drawing.draw_landmarks(
                    annotated_image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
                mp_drawing.draw_landmarks(
                    annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                # frame -> 아무것도 그려지지 않은 이미지
                # 그 위에 선이나 점같은게 그려진 이미지를 그리는 부분
                frame[start[1]:end[1], start[0]:end[0]] = annotated_image
        return frame