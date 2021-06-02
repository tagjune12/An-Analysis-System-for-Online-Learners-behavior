from PyQt5.QtWidgets import *
from PyQt5 import uic
import sys
import overlay
import threading


ui_file = uic.loadUiType('./main.ui')[0]

class MainWindow(QMainWindow, ui_file):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.olClass = overlay.OverlayClass()

        # push button 이벤트 click listener
        self.analysisStartBtn.clicked.connect(self.analysisStartBtn_clicked)
        self.analysisEndBtn.clicked.connect(self.analysisEndBtn_clicked)

    # 분석 시작 버튼 클릭 이벤트 함수
    def analysisStartBtn_clicked(self):
        print("Start Button Clicked")
        olThread = threading.Thread(target=self.olClass.WinMain())
        olThread.daemon = True
        olThread.start()
        # t = threading.Thread(target=self.olClass.SetWindow, args=(self.olClass.hWnd,))
        # t.daemon = True
        # t.start()


    # 분석 종료 버튼 클릭 이벤트 함수
    def analysisEndBtn_clicked(self):
        print("End Button Clicked")
        # self.olClass.ExitOverlay()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MainWindow()
    myWindow.show()
    app.exec_()