import os

import win32gui
import pywintypes
import multiprocessing
import time

# 부모 윈도우의 핸들을 검사합니다.
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

def producer(q, q2, semaphore=None): # q 는 공유메모리사용을 위한 queue다
    proc = multiprocessing.current_process()
    print(f'producer:{proc.name}')
    print(f'PID:{os.getpid()}')
    temp = [0,0,1,1] # 타겟윈도우의 기존 위치
    while True:
        # semaphore.acquire()
        print('트래커 세마포어 얻음')
        # print("읭")
        # tWnd = WindowFinder('Zoom 회의').GetHwnd()
        tWnd = WindowFinder('카메라').GetHwnd()
        # 타겟윈도우가 존재할경우
        if tWnd != 0 :
            print("zoom tracker if문 실행")
            tRect = win32gui.GetWindowRect(tWnd) # tuple(L,T,R,B)
            # 타겟 윈도우의 위치가 변했을 때만 창의 위치를 큐에 삽입
            # if temp[0] != tRect[0] and temp[1] != tRect[1] and temp[2] != tRect[2] and temp[3] != tRect[3]:
            temp[0] = tRect[0]
            temp[1] = tRect[1]
            temp[2] = tRect[2]
            temp[3] = tRect[3]

            q.put(tRect)
            # arr = tRect
            q2.put(tRect)

        else:
            print("zoom tracker else문 실행")
            q.put(temp)
            # q2.put(tRect)

        time.sleep(5)

        # semaphore.release()
        print('트래커 세마포어 릴리즈')