from PyQt5.QtWidgets import *
from PyQt5 import uic
import sys
import display
import threading
import screencapture
from Overlay import overlay_2

ui_file = uic.loadUiType('./UI/main.ui')[0]

class MainWindow(QMainWindow, ui_file):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # push button 이벤트 click listener
        self.analysisStartBtn.clicked.connect(self.analysisStartBtn_clicked)
        self.analysisEndBtn.clicked.connect(self.analysisEndBtn_clicked)

        self.displayclass = display.Display()
        self.overlayClass = overlay_2.Sticker('red.gif', xy=[300, 300], size=0.3, on_top=True)

    # 분석 시작 버튼 클릭 이벤트 함수
    def analysisStartBtn_clicked(self):
        print("Start Button Clicked")
        screencapture.CaptureBoard()

        self.displayclass.set_flag(True)
        analysis_thread = threading.Thread(target=self.displayclass.analysizeStart())
        # analysis_thread.daemon = True
        analysis_thread.start()

        self.overlayClass.show()
        self.overlayClass.RunSetWindow()




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