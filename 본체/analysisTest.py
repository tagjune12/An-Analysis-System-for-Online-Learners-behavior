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

# 학습자 태도 분석하는 함수 by.택준
def analysize(queue):
    proc = multiprocessing.current_process()
    # Yolo 사용하는 경우
    # peopleDetector = detect_person.PeopleDetector()
    featureDetector = detect_feature.FeatureDetector()
    # Zoom 회의 프로그램을 찾는다.
    tWnd = WindowFinder('Zoom 회의').GetHwnd()
    # tWnd = WindowFinder('카메라').GetHwnd()

    while True:
        print("분석시작")

        if tWnd != 0:
            # Zoom 회의 프로그램의 위치
            tRect = win32gui.GetWindowRect(tWnd)  # tuple(L,T,R,B)

            image = ImageGrab.grab(bbox=tRect)
            image = np.array(image)


            capture_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#---------------------------- YOLO 사용 안하는 경우 -------------------------------
            image_height, image_width = capture_frame.shape[0], capture_frame.shape[1]

            print(image_width, image_height)
            # 2인의 경우
            # image_start = [[0, 0], [int(image_width / 2), 0]] # x,y
            # image_end = [[int(image_width / 2), image_height], [image_width , image_height]] # x,y

            # 4인의 경우
            image_start = [[0, 0], [int(image_width / 2), 0], [0, int(image_height / 2)], [int(image_width / 2), int(image_height / 2)]] # x,y
            image_end = [[int(image_width / 2), int(image_height / 2)], [image_width , int(image_height / 2)], [int(image_width / 2), image_height], [image_width, image_height]] # x,y

            # 1인의 경우
            # image_start = [[0,0]]
            # image_end = [[image_width, image_height]]

            if len(np.squeeze(image_start)) != 0:
                # 이미지를 이용하여 특징점을 찾아내는 부분 by.택준
                features = featureDetector.detectFeaturePoints(capture_frame, image_start, image_end)  # 특징점 검출

                print(f'로그1:{len(features)}')

                # 검출한 특징점을 이용하여 태도를 분석하는 부분 by.택준
                result = clf.classify(features)

                print(f'로그2{result}')
                # 결과가 없는 경우 로그 출력
                if len(result) < 4:
                    print('Cannot find student')

                    queue.put(['None','None','None','None'])

                # 태도분석 결과가 존재할 경우 공유메모리 Queue에 분석 결과를 전달한다.
                else:
                    print(result)
                    queue.put(result)

        else:
            print('No image')

        time.sleep(1)

# 특정 윈도우를 찾아내기 위한 클래스 by.상민
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