# import win32gui
# import pywintypes
# import sys
# from PyQt5 import QtCore, QtWidgets, QtGui
# from PyQt5.QtGui import QMovie
#
# from time import sleep
# from multiprocessing import Process, Semaphore, shared_memory
# import numpy as np
# import threading
#
# class Sticker(QtWidgets.QMainWindow):
#
#     def __init__(self, name, semaphore, img_path, xy,size=1.0, on_top=False):
#         super(Sticker, self).__init__()
#         self.timer = QtCore.QTimer(self)
#         self.img_path = img_path
#         self.xy = xy
#         self.from_xy = xy
#         self.from_xy_diff = [0, 0]
#         self.to_xy = xy
#         self.to_xy_diff = [0, 0]
#         self.speed = 60
#         self.direction = [0, 0] # x: 0(left), 1(right), y: 0(up), 1(down)
#         self.size = size
#         self.on_top = on_top
#         self.localPos = None
#
#         self.setupUi()
#
#         self.shm_name = name
#         self.semaphore = semaphore
#
#         # self.show()
#
#     # 마우스 놓았을 때
#     def mouseReleaseEvent(self, a0: QtGui.QMouseEvent) -> None:
#         if self.to_xy_diff == [0, 0] and self.from_xy_diff == [0, 0]:
#             pass
#         else:
#             self.walk_diff(self.from_xy_diff, self.to_xy_diff, self.speed, restart=True)
#
#     # 마우스 눌렀을 때
#     def mousePressEvent(self, a0: QtGui.QMouseEvent):
#         self.localPos = a0.localPos()
#
#     # 드래그 할 때
#     def mouseMoveEvent(self, a0: QtGui.QMouseEvent):
#         self.timer.stop()
#         self.xy = [(a0.globalX() - self.localPos.x()), (a0.globalY() - self.localPos.y())]
#         self.move(*self.xy)
#
#     def walk(self, from_xy, to_xy, speed=60):
#         self.from_xy = from_xy
#         self.to_xy = to_xy
#         self.speed = speed
#
#         self.timer = QtCore.QTimer(self)
#         self.timer.timeout.connect(self.__walkHandler)
#         self.timer.start(1000 / self.speed)
#
#     # 초기 위치로부터의 상대적 거리를 이용한 walk
#     def walk_diff(self, from_xy_diff, to_xy_diff, speed=60, restart=False):
#         self.from_xy_diff = from_xy_diff
#         self.to_xy_diff = to_xy_diff
#         self.from_xy = [self.xy[0] + self.from_xy_diff[0], self.xy[1] + self.from_xy_diff[1]]
#         self.to_xy = [self.xy[0] + self.to_xy_diff[0], self.xy[1] + self.to_xy_diff[1]]
#         self.speed = speed
#         if restart:
#             self.timer.start()
#         else:
#             self.timer.timeout.connect(self.__walkHandler)
#             self.timer.start(1000 / self.speed)
#
#
#     def __walkHandler(self):
#         if self.xy[0] >= self.to_xy[0]:
#             self.direction[0] = 0
#         elif self.xy[0] < self.from_xy[0]:
#             self.direction[0] = 1
#
#         if self.direction[0] == 0:
#             self.xy[0] -= 1
#         else:
#             self.xy[0] += 1
#
#         if self.xy[1] >= self.to_xy[1]:
#             self.direction[1] = 0
#         elif self.xy[1] < self.from_xy[1]:
#             self.direction[1] = 1
#
#         if self.direction[1] == 0:
#             self.xy[1] -= 1
#         else:
#             self.xy[1] += 1
#
#         self.move(*self.xy)
#
#     def setupUi(self):
#         centralWidget = QtWidgets.QWidget(self)
#         self.setCentralWidget(centralWidget)
#
#         flags = QtCore.Qt.WindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint if self.on_top else QtCore.Qt.FramelessWindowHint)
#         self.setWindowFlags(flags)
#         self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True) ####
#         self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True) ####
#
# ##############
#         label = QtWidgets.QLabel(centralWidget)
#         movie = QMovie(self.img_path)
#         label.setMovie(movie)
#
#         label2 = QtWidgets.QLabel(centralWidget)
#         label2.move(0,540)
#         # label2.setGeometry(QtCore.QRect(0,960,200,200)) #LTWH
#         # movie2 = QMovie(self.img_path)
#         label2.setMovie(movie)
#
#         label3 = QtWidgets.QLabel(centralWidget)
#         label3.move(960, 0)
#         label3.setMovie(movie)
#         label4 = QtWidgets.QLabel(centralWidget)
#         label4.move(960, 540)
#         label4.setMovie(movie)
#
#
#         movie.start()
#         movie.stop()
#
#         w = int(movie.frameRect().size().width() * self.size)
#         h = int(movie.frameRect().size().height() * self.size)
#         movie.setScaledSize(QtCore.QSize(w, h))
#         movie.start()
#         # label2.move(self.xy[0]+h, self.xy[1]+w)#################
#         # movie2.setScaledSize(QtCore.QSize(w, h))
#         # movie2.start()
#
#         ##창크기
#         # self.setGeometry(self.xy[0], self.xy[1], 1920, 1080)###############
#         # print(self.xy)
#
#     def SetWindow(self, a, b):
#         # while True:
#             # tWnd = WindowFinder('계산기').GetHwnd()
#             # tWnd = WindowFinder('Zoom 회의').GetHwnd()
#             tWnd = WindowFinder('카메라').GetHwnd()
#             # print(tWnd) #test
#             if tWnd != 0:
#                 self.semaphore.acquire()
#                 exist_shm = shared_memory.SharedMemory(name=self.shm_name)
#
#                 tRect = win32gui.GetWindowRect(tWnd)  # tuple(L,T,R,B)
#                 wWidth = tRect[2] - tRect[0]
#                 wHeight = tRect[3] - tRect[1]
#                 self.setGeometry(tRect[0], tRect[1], wWidth, wHeight)
#
#                 # 추가한 부분 by.이택준
#                 print(f'Log from SetWindow:{tRect}')
#
#                 window_size = np.ndarray((4,), dtype=np.int32, buffer=exist_shm.buf)
#                 window_size[0] = tRect[0]
#                 window_size[1] = tRect[1]
#                 window_size[2] = tRect[2]
#                 window_size[3] = tRect[3]
#                 self.semaphore.release()
#
#     def RunSetWindow(self):
#         self.show()
#         self.timer = QtCore.QTimer(self)
#         self.timer.setInterval(100)
#         self.timer.timeout.connect(self.SetWindow)
#         self.timer.start()
#         # threading.Timer(0.05,self.SetWindow, args=(capture_size, shm, semaphore)).start()
#
#
#
#     # def SetWindow(self):
#     #     # while True:
#     #         # tWnd = WindowFinder('계산기').GetHwnd()
#     #         # tWnd = WindowFinder('Zoom 회의').GetHwnd()
#     #         tWnd = WindowFinder('카메라').GetHwnd()
#     #         # print(tWnd) #test
#     #         if tWnd != 0:
#     #             tRect = win32gui.GetWindowRect(tWnd)  # tuple(L,T,R,B)
#     #             wWidth = tRect[2] - tRect[0]
#     #             wHeight = tRect[3] - tRect[1]
#     #             self.setGeometry(tRect[0], tRect[1], wWidth, wHeight)
#     #             self.show()
#     #             # 추가한 부분 by.이택준
#     #             # screencapture.CaptureBoard().set_capture_size(tRect)
#     #             print(f'Log from SetWindow:{tRect}')
#     #
#     #         # sleep(0.1)
#     #
#     # def RunSetWindow(self):
#     #     self.timer = QtCore.QTimer(self)
#     #     self.timer.setInterval(100)
#     #     self.timer.timeout.connect(self.SetWindow)
#     #     self.timer.start()
#
# '''
#     # 추가한 부분 by.이택준
#     def execute_overlay(self):
#         # queue = Queue()
#         # value = Array('i',[])
#         # work2 = Process(target=self.RunSetWindow,name='오버레이 모듈',args=(value))
#         # work2.start()
# '''
#
# # 부모 윈도우의 핸들을 검사합니다.
# class WindowFinder:
#     def __init__(self, windowname):
#         try:
#             win32gui.EnumWindows(self.__EnumWindowsHandler, windowname)
#         except pywintypes.error as e:
#             # 발생된 예외 중 e[0]가 0이면 callback이 멈춘 정상 케이스
#             if e == 0: pass
#
#     def __EnumWindowsHandler(self, hwnd, extra):
#         wintext = win32gui.GetWindowText(hwnd)
#         if wintext.find(extra) != -1:
#             self.__hwnd = hwnd
#             return pywintypes.FALSE  # FALSE는 예외를 발생시킵니다.
#
#     def GetHwnd(self):
#         return self.__hwnd
#
#     __hwnd = 0
#
#
#
#
import win32gui
import pywintypes
import sys
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtGui import QMovie


class Sticker(QtWidgets.QMainWindow):

    def __init__(self, img_path, xy, size=1.0, on_top=False):
        super(Sticker, self).__init__()
        self.timer = QtCore.QTimer(self)
        self.img_path = img_path
        self.xy = xy
        self.from_xy = xy
        self.from_xy_diff = [0, 0]
        self.to_xy = xy
        self.to_xy_diff = [0, 0]
        self.speed = 60
        self.direction = [0, 0] # x: 0(left), 1(right), y: 0(up), 1(down)
        self.size = size
        self.on_top = on_top
        self.localPos = None

        self.setupUi()
        # self.show()

        self.window_size = [0, 0, 0, 0]


    def mouseReleaseEvent(self, a0: QtGui.QMouseEvent) -> None:
        if self.to_xy_diff == [0, 0] and self.from_xy_diff == [0, 0]:
            pass
        else:
            self.walk_diff(self.from_xy_diff, self.to_xy_diff, self.speed, restart=True)

    # 마우스 눌렀을 때
    def mousePressEvent(self, a0: QtGui.QMouseEvent):
        self.localPos = a0.localPos()

    # 드래그 할 때
    def mouseMoveEvent(self, a0: QtGui.QMouseEvent):
        self.timer.stop()
        self.xy = [(a0.globalX() - self.localPos.x()), (a0.globalY() - self.localPos.y())]
        self.move(*self.xy)

    def walk(self, from_xy, to_xy, speed=60):
        self.from_xy = from_xy
        self.to_xy = to_xy
        self.speed = speed

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.__walkHandler)
        self.timer.start(1000 / self.speed)

    # 초기 위치로부터의 상대적 거리를 이용한 walk
    def walk_diff(self, from_xy_diff, to_xy_diff, speed=60, restart=False):
        self.from_xy_diff = from_xy_diff
        self.to_xy_diff = to_xy_diff
        self.from_xy = [self.xy[0] + self.from_xy_diff[0], self.xy[1] + self.from_xy_diff[1]]
        self.to_xy = [self.xy[0] + self.to_xy_diff[0], self.xy[1] + self.to_xy_diff[1]]
        self.speed = speed
        if restart:
            self.timer.start()
        else:
            self.timer.timeout.connect(self.__walkHandler)
            self.timer.start(1000 / self.speed)



    def __walkHandler(self):
        if self.xy[0] >= self.to_xy[0]:
            self.direction[0] = 0
        elif self.xy[0] < self.from_xy[0]:
            self.direction[0] = 1

        if self.direction[0] == 0:
            self.xy[0] -= 1
        else:
            self.xy[0] += 1

        if self.xy[1] >= self.to_xy[1]:
            self.direction[1] = 0
        elif self.xy[1] < self.from_xy[1]:
            self.direction[1] = 1

        if self.direction[1] == 0:
            self.xy[1] -= 1
        else:
            self.xy[1] += 1

        self.move(*self.xy)

    def setupUi(self):
        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)

        flags = QtCore.Qt.WindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint if self.on_top else QtCore.Qt.FramelessWindowHint)
        self.setWindowFlags(flags)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True) ####
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True) ####

##############
        label = QtWidgets.QLabel(centralWidget)
        movie = QMovie(self.img_path)
        label.setMovie(movie)

        label2 = QtWidgets.QLabel(centralWidget)
        label2.move(0,540)

        label2.setMovie(movie)

        label3 = QtWidgets.QLabel(centralWidget)
        label3.move(960, 0)
        label3.setMovie(movie)
        label4 = QtWidgets.QLabel(centralWidget)
        label4.move(960, 540)
        label4.setMovie(movie)


        movie.start()
        movie.stop()

        w = int(movie.frameRect().size().width() * self.size)
        h = int(movie.frameRect().size().height() * self.size)
        movie.setScaledSize(QtCore.QSize(w, h))
        movie.start()

    def SetWindow(self):
        # tWnd = WindowFinder('계산기').GetHwnd()
        tWnd = WindowFinder('카메라').GetHwnd()
        # print(tWnd) #test
        if tWnd != 0 :
            tRect = win32gui.GetWindowRect(tWnd) # tuple(L,T,R,B)
            wWidth = tRect[2] - tRect[0]
            wHeight = tRect[3] - tRect[1]
            self.setGeometry(tRect[0], tRect[1], wWidth, wHeight)
            self.window_size[0] = tRect[0]
            self.window_size[1] = tRect[1]
            self.window_size[2] = tRect[2]
            self.window_size[3] = tRect[3]

    def RunOverlay(self):
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.SetWindow)
        self.timer.start()
        self.show()
    def StopOverlay(self):
        self.timer.stop()
        self.hide()

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
