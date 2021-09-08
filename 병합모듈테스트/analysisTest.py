import multiprocessing
import os
import numpy as np
import cv2
import pywintypes
import win32gui
from PIL import ImageGrab

import classification_module as clf
import detect_feature
import detect_person
import time

def analysize(queue):
    print('으엙')
    proc = multiprocessing.current_process()
    print(f'분석모듀우우우우ㅜㄹ:{proc.name}')
    print(f'PID:{os.getpid()}')

    peopleDetector = detect_person.PeopleDetector()
    featureDetector = detect_feature.FeatureDetector()

    tWnd = WindowFinder('Zoom 회의').GetHwnd()

    while True:
        print("분석시작")

        if tWnd != 0:
            tRect = win32gui.GetWindowRect(tWnd)  # tuple(L,T,R,B)

            image = ImageGrab.grab(bbox=tRect)
            image = np.array(image)

            capture_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            flag, image_start, image_end = peopleDetector.detect(capture_frame)  # 사람 감지

            if len(np.squeeze(image_start)) != 0:
                features = featureDetector.detectFeaturePoints(capture_frame, image_start, image_end)  # 특징점 검출
                result = clf.classify(features)

                # 화면에 표시
                if result == None:
                    print('Cannot find student')


                else:
                    a = [result]
                    print(a)
                    queue.put(a)
                    # print(result)

        else:
            print('No image')
            # continue
            # break
        time.sleep(1)


class WindowFinder:
    def __init__(self, windowname):
        try:
            win32gui.EnumWindows(self.__EnumWindowsHandler, windowname)
        except pywintypes.error as e:
            # 발생된 예외 중 e[0]가 0이면 callback이 멈춘 정상 케이스
            if e == 0: pass

    def __EnumWindowsHandler(self, hwnd, extra):
        wintext = win32gui.GetWindowText(hwnd)
        if wintext.find(extra) != -1:
            self.__hwnd = hwnd
            return pywintypes.FALSE  # FALSE는 예외를 발생시킵니다.

    def GetHwnd(self):
        return self.__hwnd

    __hwnd = 0