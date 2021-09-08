import multiprocessing
import os
import time

from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5 import uic
import sys

from multiprocessing import Queue, Process
from Overlay import overlay_2
import analysisTest

ui_file = uic.loadUiType('./UI/main.ui')[0]
queue = Queue()
# semaphore = Semaphore(2)

class MainWindow(QMainWindow, ui_file):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.overlay = overlay_2.Sticker(queue, './Overlay/red.gif', xy=[300, 300], size=0.3, on_top=True)

        # push button 이벤트 click listener
        self.analysisStartBtn.clicked.connect(self.analysisStartBtn_clicked)
        self.analysisEndBtn.clicked.connect(self.analysisEndBtn_clicked)

    # 분석 시작 버튼 클릭 이벤트 함수
    def analysisStartBtn_clicked(self):
        print("Start Button Clicked")

        self.overlay.RunOverlay()
        p = Process(name='analysizer', target=analysisTest.analysize, args=(queue,), daemon=True)
        p.start()


    # 분석 종료 버튼 클릭 이벤트 함수
    def analysisEndBtn_clicked(self):
        print("End Button Clicked")



if __name__ == "__main__":
    proc = multiprocessing.current_process()
    print(f'__main__:{proc.name}')
    print(f'PID:{os.getpid()}')

    app = QApplication(sys.argv)

    # overlay = overlay_2.Sticker(queue1, './Overlay/red.gif', xy=[300, 300], size=0.3, on_top=True)

    myWindow = MainWindow()
    myWindow.show()
    app.exec_()