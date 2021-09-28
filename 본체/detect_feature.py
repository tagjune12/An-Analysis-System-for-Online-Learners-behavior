import mediapipe as mp
import cv2



class FeatureDetector:
    def __init__(self):
        # self.mp_drawing = mp.solutions.drawing_utils
        # self.mp_drawing = drawing
        self.mp_holistic = mp.solutions.holistic
    def detectFeaturePoints(self, frame, image_start:list, image_end:list):
        # mp_drawing = self.mp_drawing
        mp_holistic = self.mp_holistic
        results = ['None','None','None','None']
        for i in range(len(image_start)):
            print('Log from detectFeaturePoints')
            start = tuple(image_start[i])
            end = tuple(image_end[i])
            # image -> human 영역만 crop 된 이미지
            image = frame[start[1]:end[1], start[0]:end[0]]  # 이미지[Y좌표, X좌표]
            # Convert the BGR image to RGB before processing.
            with mp_holistic.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.5) as holistic:
                # results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                # results.append(holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
                results[i] = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))



        # return frame
        return results # list[[landmark]]
