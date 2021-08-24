from PyQt5.QtWidgets import *
from PyQt5 import uic
import sys
import display

import screencapture
from Overlay import overlay_2
import multiprocessing
from multiprocessing import Process, Semaphore, shared_memory
import numpy as np
import time

ui_file = uic.loadUiType('./UI/main.ui')[0]


class MainWindow(QMainWindow, ui_file):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # push button 이벤트 click listener
        self.analysisStartBtn.clicked.connect(self.analysisStartBtn_clicked)
        self.analysisEndBtn.clicked.connect(self.analysisEndBtn_clicked)

        semaphore = Semaphore(1)
        capture_size = (1,1,1,1)
        capture_size = np.array(capture_size)
        self.shm = shared_memory.SharedMemory(create=True, size= capture_size.nbytes)


        # self.overlayClass = overlay_2.Sticker('red.gif', xy=[300, 300], size=0.3, on_top=True)
        self.overlayClass = overlay_2.Sticker(self.shm.name, semaphore,'red.gif', xy=[300, 300],size=0.3, on_top=True)
        self.displayclass = display.Display(self.shm.name, semaphore)

    # 분석 시작 버튼 클릭 이벤트 함수
    def analysisStartBtn_clicked(self):
        print("Start Button Clicked")
        screencapture.CaptureBoard()

        self.displayclass.set_flag(True)
        # analysis_thread = threading.Thread(target=self.displayclass.analysize())
        # analysis_thread.daemon = True
        # analysis_thread.start()


        # self.overlayClass.show()
        # self.overlayClass.RunSetWindow()

        # self.overlayClass.execute_overlay()
        # self.displayclass.start_analysis()

        # semaphore = Semaphore(1)
        # # capture_size = screencapture.CaptureBoard().get_capture_size()
        # capture_size = (1,1,1,1)
        # capture_size = np.array(capture_size)
        # shm = shared_memory.SharedMemory(create=True, size= capture_size.nbytes)
        #
        # shared_data = np.ndarray(capture_size.shape, dtype=capture_size.dtype, buffer=shm.buf)



        #-------------공유 메모리------------------
        # work1 = Process(target=self.overlayClass.SetWindow(shared_data, shm.name, semaphore))
        # work2 = Process(target=self.displayclass.analysize(shared_data, shm.name, semaphore))
        #
        #
        # work1.start()
        # work2.start()


        #------------프로세스 풀------------
        pool = multiprocessing.Pool(processes=2)
        pool.apply(self.overlayClass.RunSetWindow())
        pool.apply(self.displayclass.analysize())



        # work1 = Process(target=self.overlayClass.SetWindow)

        # work1 = Process(target=self.overlayClass.RunSetWindow())
        # work2 = Process(target=self.displayclass.analysize())

        # work1.start()
        # work2.start()



    # 분석 종료 버튼 클릭 이벤트 함수
    def analysisEndBtn_clicked(self):
        print("End Button Clicked")
        self.displayclass.set_flag(False)

        self.overlayClass.timer.stop()
        self.overlayClass.hide()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MainWindow()
    myWindow.show()
    app.exec_()