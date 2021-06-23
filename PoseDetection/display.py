import cv2
import numpy as np
import time
import detect_feature
import detect_person
import screencapture

class Display:
    def __init__(self):
        self.check = True

    def analysizeStart(self):
        captureBoard = screencapture.CaptureBoard()
        peopleDetector = detect_person.PeopleDetector()
        featureDetector = detect_feature.FeatureDetector()
        start = time.time()
        frame_count = 0
        while True:
            frame_count += 1
            capture_frame = captureBoard.captureScreen()  # 화면 캡처
            capture_frame = cv2.cvtColor(capture_frame, cv2.COLOR_BGR2RGB)
            flag, image_start, image_end = peopleDetector.detect(capture_frame)  # 사람 감지

            if flag and len(np.squeeze(image_start)) != 0:
                capture_frame = featureDetector.detectFeaturePoints(capture_frame, image_start, image_end)  # 특징점 검출

            # 화면에 표시
            capture_frame = cv2.resize(capture_frame, (960, 540))
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("result", capture_frame)
            end = time.time()
            print(f"Time Lapse: {frame_count / (end - start)} ")
            # if cv2.waitKey(1) & 0xFF == ord('q'): break
            if cv2.waitKey(1) :
                if not self.check:
                    print("Flag: ", self.check)
                    cv2.destroyWindow("result")
                    break



    def set_flag(self, arg):
        self.check = arg