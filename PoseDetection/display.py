import cv2
import numpy as np
import time

import detect_person
import detect_feature
import screencapture

def main():

    captureBoard = screencapture.CaptureBoard()
    peopleDetector = detect_person.PeopleDetector()
    featureDetector = detect_feature.FeatureDetector()
    start = time.time()
    frame_count = 0
    while True:
        frame_count+=1
        capture_frame = captureBoard.captureScreen() # 화면 캡처
        capture_frame = cv2.cvtColor(capture_frame,cv2.COLOR_BGR2RGB)



        flag, image_start, image_end = peopleDetector.detect(capture_frame) # 사람 감지


        if flag and len(np.squeeze(image_start)) != 0:
            capture_frame = featureDetector.detectFeaturePoints(capture_frame, image_start, image_end) # 특징점 검출


        #화면에 표시
        capture_frame = cv2.resize(capture_frame,(960,540))
        cv2.namedWindow("result",cv2.WINDOW_AUTOSIZE)
        cv2.imshow("result", capture_frame)
        end = time.time()
        print(f"Time Lapse: {frame_count/(end - start)} ")
        if cv2.waitKey(1) & 0xFF == ord('q'): break






if __name__ == '__main__':
    try:
        # app.run(main)
        main()
    except SystemExit:
        pass

'''
    start = time.time()
    end = time.time()
    print(f"Time Lapse: {(end-start)/1000} ")
'''