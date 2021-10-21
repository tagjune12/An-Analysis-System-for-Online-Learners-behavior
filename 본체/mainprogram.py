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

# UI파일 불러오는 부분
ui_file = uic.loadUiType('./UI/main2.ui')[0]
# 멀티 프로세싱과 IPC를 위한 공유 메모리 Queue by.택준
queue = Queue()


class MainWindow(QMainWindow, ui_file):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.overlay = overlay_2.Sticker(queue, './Overlay/red.gif', xy=[300, 300], size=0.3, on_top=True)

        # button과 click listener 연결 by.택준
        self.analysisStartBtn.clicked.connect(self.analysisStartBtn_clicked)
        self.analysisEndBtn.clicked.connect(self.analysisEndBtn_clicked)

        self.process = None

    # 분석 시작 버튼 클릭 이벤트 함수 by.택준
    def analysisStartBtn_clicked(self):
        print("Start Button Clicked")

        self.overlay.RunOverlay()
        # 태도분석 기능을 멀티프로세스로 실행 by.택준


        if self.process is None:
            self.process = Process(name='analysizer', target=analysisTest.analysize, args=(queue,), daemon=True)
            self.process.start()


    # 분석 종료 버튼 클릭 이벤트 함수
    def analysisEndBtn_clicked(self):
        print("End Button Clicked")
        self.overlay.StopOverlay()


if __name__ == "__main__":
    proc = multiprocessing.current_process()
    print(f'__main__:{proc.name}')
    print(f'PID:{os.getpid()}')

    app = QApplication(sys.argv)

    myWindow = MainWindow()
    myWindow.show()
    app.exec_()