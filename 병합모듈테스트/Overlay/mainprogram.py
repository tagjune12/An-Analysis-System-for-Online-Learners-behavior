from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import *
from PyQt5 import uic
import sys
import overlay_2
import threading
import screencapture


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

        screencapture.CaptureBoard()
        self.overlayClass.show()
        self.overlayClass.RunSetWindow()


    # 분석 종료 버튼 클릭 이벤트 함수
    def analysisEndBtn_clicked(self):
        print("End Button Clicked")
        self.overlayClass.timer.stop()
        self.overlayClass.hide()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MainWindow()
    myWindow.show()
    app.exec_()