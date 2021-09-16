import multiprocessing
import os
import sys
import time
import win32gui
import pywintypes

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QMovie


class Sticker(QtWidgets.QMainWindow):

    def __init__(self, queue, img_path, xy, size=1.0, on_top=False):
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
        self.queue = queue
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
        proc = multiprocessing.current_process()
        # print(f'SetWindow:{proc.name}')
        # print(f'PID:{os.getpid()}')
        tWnd = WindowFinder('Zoom 회의').GetHwnd()
        # tWnd = WindowFinder('카메라').GetHwnd()
        # print(tWnd) #test
        if tWnd != 0 :
            tRect = win32gui.GetWindowRect(tWnd) # tuple(L,T,R,B)
            wWidth = tRect[2] - tRect[0]
            wHeight = tRect[3] - tRect[1]
            self.setGeometry(tRect[0], tRect[1], wWidth, wHeight)


    def RunOverlay(self):
        # proc = multiprocessing.current_process()
        # print(f'RunOverlay:{proc.name}')
        # print(f'PID:{os.getpid()}')


        self.timer = QtCore.QTimer(self)
        # self.timer.setInterval(100)
        self.timer.timeout.connect(self.SetWindow)
        self.timer.start()
        self.show()

        self.consumer = Consumer(self.queue)
        self.consumer.poped.connect(self.get_state_result)
        self.consumer.start()


    def StopOverlay(self):
        self.timer.stop()
        self.hide()

    # 분석결과 얻어와서 결과에 따라 이미지 뿌려주는거
    def get_state_result(self, value):
        print(f'분석결과 출력 in Overlay:{value}')

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

# 태도분석 결과 받으면 알려주는함수 구현해야함
class Consumer(QThread):
    poped = pyqtSignal(list)

    def __init__(self, q):
        super().__init__()
        self.q = q
        # self.semaphore = sem

    def run(self):
        print("consumer 쓰레드 실행")
        while True:

            if not self.q.empty():
                # self.semaphore.acquire()
                data = self.q.get()

                self.poped.emit(data)
                print('시그널 보냄')

            # time.sleep(10)