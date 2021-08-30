import win32gui
import win32api
import win32con
import threading

class OverlayClass():
    def __init__(self):
        self.szClassName = 'MyFirstProgram'
        self.TargetName = '계산기'
        self.flag = False
        self.hWnd = None

    def WinMain(self):

        hInstance = win32api.GetModuleHandle()

        wndClass = win32gui.WNDCLASS()
        wndClass.hInstance = hInstance
        wndClass.lpszClassName = self.szClassName
        wndClass.lpfnWndProc = self.WndProc
        wndClass.style = win32con.CS_DBLCLKS
        wndClass.hIcon = win32gui.LoadIcon(0, win32con.IDI_APPLICATION)
        wndClass.hCursor = win32gui.LoadCursor(0, win32con.IDC_ARROW)
        wndClass.cbWndExtra = 0
        wndClass.hbrBackground = win32gui.GetStockObject(win32con.BLACK_BRUSH)

        tWnd = win32gui.FindWindow(0, self.TargetName)
        if tWnd == 0:
            exit(1)     # 시작시 타켓윈도우 없으면 종료
        # while tWnd == 0:
        #     tWnd = win32gui.FindWindow(0, self.TargetName) # 타겟윈도우 찾을때까지 반복
        tRect = win32gui.GetWindowRect(tWnd)
        wWidth = tRect[2] - tRect[0]
        wHeight = tRect[3] - tRect[1]
        if not tWnd :
            wWidth = 400
            wHeight = 350

        wndClassAtom = None
        try:
            wndClassAtom = win32gui.RegisterClass(wndClass)
        except Exception as e:
            print(e)
            raise e
        self.hWnd = win32gui.CreateWindowEx(win32con.WS_EX_TOPMOST | win32con.WS_EX_TRANSPARENT, wndClassAtom, "Overlay",
            win32con.WS_POPUP, 100, 90, wWidth, wHeight, 0, 0, hInstance, None)
        #hWnd

        # Show & update the window
        win32gui.ShowWindow(self.hWnd, win32con.SW_SHOW)
        win32gui.UpdateWindow(self.hWnd)

        win32gui.SetWindowLong(self.hWnd, win32con.GWL_EXSTYLE, win32gui.GetWindowLong(self.hWnd, win32con.GWL_EXSTYLE) ^ win32con.WS_EX_LAYERED)
        win32gui.SetLayeredWindowAttributes(self.hWnd, win32api.RGB(0, 0, 0), 0, win32con.LWA_COLORKEY)

        # win32ui.CreateThread(SetWindow, hWnd)
        t = threading.Thread(target=self.SetWindow, args=(self.hWnd,))
        t.daemon = True
        t.start()

    #msg pump
        msg = win32gui.GetMessage(self.hWnd, 0, 0)
        msg = msg[1]

        while msg[1]:
            win32gui.TranslateMessage(msg)
            win32gui.DispatchMessage(msg)
            msg = win32gui.GetMessage(self.hWnd, 0, 0)
            msg = msg[1]
            if self.flag == True:
                exit(0)
        # win32gui.PumpMessages()

    def WndProc(self, hWnd, message, wParam, lParam):
        if message == win32con.WM_PAINT:
            rect = win32gui.GetClientRect(hWnd) # tuple(L,T,R,B)
            hDC, paintStruct = win32gui.BeginPaint(hWnd)
            win32gui.SetTextColor(hDC, win32api.RGB(255, 0, 0))
            win32gui.SetBkMode(hDC, 1)
            rect2 = (rect[0] + 10, rect[1] + 10, rect[2], rect[3])
            rect3 = (rect[0] + 10, rect[1] + 30, rect[2], rect[3])
            win32gui.DrawText(hDC,'Overlay Test', -1, rect2, win32con.DT_SINGLELINE | win32con.DT_NOCLIP)
            win32gui.DrawText(hDC, 'Test message2', -1, rect3, win32con.DT_SINGLELINE | win32con.DT_NOCLIP)
            ###
            memDC = win32gui.CreateCompatibleDC(hDC)
            image1 = win32gui.LoadImage(None, "1111.bmp", win32con.IMAGE_BITMAP, 0, 0, win32con.LR_LOADFROMFILE)
            OldBitmap = win32gui.SelectObject(memDC, image1)

            if image1 == 0:
                print("image load error")
                exit(1)
            win32gui.BitBlt(hDC, 10, 50, 100, 100, memDC, 0, 0, win32con.SRCCOPY)
            win32gui.SelectObject(memDC, OldBitmap)
            win32gui.DeleteObject(image1)
            win32gui.DeleteObject(memDC)

            win32gui.EndPaint(hWnd, paintStruct)

        elif message == win32con.WM_DESTROY:
            print("Being destroyed")
            win32gui.PostQuitMessage(0)


        else:
            return win32gui.DefWindowProc(hWnd, message, wParam, lParam)

    def SetWindow(self, hwnd):
        while True :
            tWnd = win32gui.FindWindow(0, self.TargetName) # 타겟이름이랑 같은 윈도우를 찾음
            print(tWnd) #test
            if tWnd : # 찾은경우
                tRect = win32gui.GetWindowRect(tWnd)
                wWidth = tRect[2] - tRect[0]
                wHeight = tRect[3] - tRect[1]
                dwStyle = win32gui.GetWindowLong(tWnd, win32con.GWL_STYLE) #윈도우의 길이를 가져옴
                if (dwStyle & win32con.WS_BORDER) : # WS_BORDER속성이 있으면 세로길이 보정
                    tRect2 = (tRect[0], tRect[1] + 23, tRect[2], tRect[3])
                    wHeight -= 23
                win32gui.MoveWindow(hwnd, tRect2[0], tRect2[1], wWidth, wHeight, True)
            else : #못찾은 경우
                print("setwindow exit") ##
                # win32gui.DestroyWindow(hwnd)
                self.flag = True
                break

    def ExitOverlay(self):
        # win32gui.DestroyWindow(self.hwnd)
        # win32gui.PostQuitMessage(0)
        print("Exit Overlay")

# class OverlayClass():
#     def __init__(self):
#         self.szClassName = 'MyFirstProgram'
#         self.TargetName = '계산기'
#         self.flag = False
#         self.hWnd = None
#
#     def WinMain(self):
#
#         hInstance = win32api.GetModuleHandle()
#
#         wndClass = win32gui.WNDCLASS()
#         wndClass.hInstance = hInstance
#         wndClass.lpszClassName = self.szClassName
#         wndClass.lpfnWndProc = self.WndProc
#         wndClass.style = win32con.CS_DBLCLKS
#         wndClass.hIcon = win32gui.LoadIcon(0, win32con.IDI_APPLICATION)
#         wndClass.hCursor = win32gui.LoadCursor(0, win32con.IDC_ARROW)
#         wndClass.cbWndExtra = 0
#         wndClass.hbrBackground = win32gui.GetStockObject(win32con.BLACK_BRUSH)
#
#         tWnd = win32gui.FindWindow(0, self.TargetName)
#         if tWnd == 0:
#             exit(1)     # 시작시 타켓윈도우 없으면 종료
#         # while tWnd == 0:
#         #     tWnd = win32gui.FindWindow(0, self.TargetName) # 타겟윈도우 찾을때까지 반복
#         tRect = win32gui.GetWindowRect(tWnd)
#         wWidth = tRect[2] - tRect[0]
#         wHeight = tRect[3] - tRect[1]
#         if not tWnd :
#             wWidth = 400
#             wHeight = 350
#
#         wndClassAtom = None
#         try:
#             wndClassAtom = win32gui.RegisterClass(wndClass)
#         except Exception as e:
#             print(e)
#             raise e
#         self.hWnd = win32gui.CreateWindowEx(win32con.WS_EX_TOPMOST | win32con.WS_EX_TRANSPARENT, wndClassAtom, "Overlay",
#             win32con.WS_POPUP, 100, 90, wWidth, wHeight, 0, 0, hInstance, None)
#         #hWnd
#
#         # Show & update the window
#         win32gui.ShowWindow(self.hWnd, win32con.SW_SHOW)
#         win32gui.UpdateWindow(self.hWnd)
#
#         win32gui.SetWindowLong(self.hWnd, win32con.GWL_EXSTYLE, win32gui.GetWindowLong(self.hWnd, win32con.GWL_EXSTYLE) ^ win32con.WS_EX_LAYERED)
#         win32gui.SetLayeredWindowAttributes(self.hWnd, win32api.RGB(0, 0, 0), 0, win32con.LWA_COLORKEY)
#
#         # win32ui.CreateThread(SetWindow, hWnd)
#         t = threading.Thread(target=self.SetWindow, args=(self.hWnd,))
#         t.daemon = True
#         t.start()
#
#     #msg pump
#         msg = win32gui.GetMessage(self.hWnd, 0, 0)
#         msg = msg[1]
#
#         while msg[1]:
#             win32gui.TranslateMessage(msg)
#             win32gui.DispatchMessage(msg)
#             msg = win32gui.GetMessage(self.hWnd, 0, 0)
#             msg = msg[1]
#             if self.flag == True:
#                 exit(0)
#         # win32gui.PumpMessages()
#
#     def WndProc(self, hWnd, message, wParam, lParam):
#         if message == win32con.WM_PAINT:
#             rect = win32gui.GetClientRect(hWnd) # tuple(L,T,R,B)
#             hDC, paintStruct = win32gui.BeginPaint(hWnd)
#             win32gui.SetTextColor(hDC, win32api.RGB(255, 0, 0))
#             win32gui.SetBkMode(hDC, 1)
#             rect2 = (rect[0] + 10, rect[1] + 10, rect[2], rect[3])
#             rect3 = (rect[0] + 10, rect[1] + 30, rect[2], rect[3])
#             win32gui.DrawText(hDC,'Overlay Test', -1, rect2, win32con.DT_SINGLELINE | win32con.DT_NOCLIP)
#             win32gui.DrawText(hDC, 'Test message2', -1, rect3, win32con.DT_SINGLELINE | win32con.DT_NOCLIP)
#             ###
#             memDC = win32gui.CreateCompatibleDC(hDC)
#             image1 = win32gui.LoadImage(None, "1111.bmp", win32con.IMAGE_BITMAP, 0, 0, win32con.LR_LOADFROMFILE)
#             OldBitmap = win32gui.SelectObject(memDC, image1)
#
#             if image1 == 0:
#                 print("image load error")
#                 exit(1)
#             win32gui.BitBlt(hDC, 10, 50, 100, 100, memDC, 0, 0, win32con.SRCCOPY)
#             win32gui.SelectObject(memDC, OldBitmap)
#             win32gui.DeleteObject(image1)
#             win32gui.DeleteObject(memDC)
#
#             win32gui.EndPaint(hWnd, paintStruct)
#
#         elif message == win32con.WM_DESTROY:
#             print("Being destroyed")
#             win32gui.PostQuitMessage(0)
#
#
#         else:
#             return win32gui.DefWindowProc(hWnd, message, wParam, lParam)
#
#
#
#     def SetWindow(self, hwnd):
#         while True :
#             tWnd = win32gui.FindWindow(0, self.TargetName) # 타겟이름이랑 같은 윈도우를 찾음
#             print(tWnd) #test
#             if tWnd : # 찾은경우
#                 tRect = win32gui.GetWindowRect(tWnd)
#                 wWidth = tRect[2] - tRect[0]
#                 wHeight = tRect[3] - tRect[1]
#                 dwStyle = win32gui.GetWindowLong(tWnd, win32con.GWL_STYLE) #윈도우의 길이를 가져옴
#                 if (dwStyle & win32con.WS_BORDER) : # WS_BORDER속성이 있으면 세로길이 보정
#                     tRect2 = (tRect[0], tRect[1] + 23, tRect[2], tRect[3])
#                     wHeight -= 23
#                 win32gui.MoveWindow(hwnd, tRect2[0], tRect2[1], wWidth, wHeight, True)
#             else : #못찾은 경우
#                 print("setwindow exit") ##
#                 # win32gui.DestroyWindow(hwnd)
#                 self.flag = True
#                 break
#
#     def ExitOverlay(self):
#         # win32gui.DestroyWindow(self.hwnd)
#         # win32gui.PostQuitMessage(0)
#         print("Exit Overlay")

# def Test():
#     while True:
#         print("Test")
#
#
# if __name__=="__main__":
#     WinMain()