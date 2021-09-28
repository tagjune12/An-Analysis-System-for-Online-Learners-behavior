from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5 import uic
import sys
import overlay_2
import threading



ui_file = uic.loadUiType('./main.ui')[0]


class MainWindow(QMainWindow, ui_file):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.overlayClass = overlay_2.Sticker('red.gif', xy=[300, 300], size=0.3, on_top=True)

        # push button 이벤트 click listener
        self.analysisStartBtn.clicked.connect(self.analysisStartBtn_clicked)
        self.analysisEndBtn.clicked.connect(self.analysisEndBtn_clicked)

    # 분석 시작 버튼 클릭 이벤트 함수
    def analysisStartBtn_clicked(self):
        print("Start Button Clicked")
        self.overlayClass.RunOverlay()
        self.RunTossWindowSize()


    # 분석 종료 버튼 클릭 이벤트 함수
    def analysisEndBtn_clicked(self):
        print("End Button Clicked")
        self.overlayClass.StopOverlay()

    def TossWindowSize(self):
        # self.클래스명.SetCaptureSize(self.overlayClass.window_size)
        print("test")

    def RunTossWindowSize(self):
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.TossWindowSize)
        self.timer.start()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MainWindow()
    myWindow.show()
    app.exec_()