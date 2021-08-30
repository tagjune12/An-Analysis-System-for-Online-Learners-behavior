from PyQt5.QtWidgets import *
from PyQt5 import uic
import sys
import display
import threading
# import overlay

ui_file = uic.loadUiType('./UI/main.ui')[0]

class MainWindow(QMainWindow, ui_file):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # push button 이벤트 click listener
        self.analysisStartBtn.clicked.connect(self.analysisStartBtn_clicked)
        self.analysisEndBtn.clicked.connect(self.analysisEndBtn_clicked)

        self.displayclass = display.Display()
        # self.olClass = overlay.OverlayClass()

    # 분석 시작 버튼 클릭 이벤트 함수
    def analysisStartBtn_clicked(self):
        print("Start Button Clicked")
        self.displayclass.set_flag(True)
        analysis_thread = threading.Thread(target=self.displayclass.analysizeStart())
        analysis_thread.daemon = True
        analysis_thread.start()

        # olThread = threading.Thread(target=self.olClass.WinMain())
        # olThread.daemon = True
        # olThread.start()


    # 분석 종료 버튼 클릭 이벤트 함수
    def analysisEndBtn_clicked(self):
        print("End Button Clicked")
        self.displayclass.set_flag(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MainWindow()
    myWindow.show()
    app.exec_()