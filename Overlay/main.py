import win32gui
import win32api
import win32con
import threading


szClassName = 'MyFirstProgram'
TargetName = '계산기'
flag = 0
# hInstance = win32api.GetModuleHandle()

def WinMain(): #hInstance, hPrevInstance, lpCmdLine, nCmdShow):

    # create and initialize window class
    # wndClass = win32gui.WNDCLASS()
    # wndClass.style = win32con.CS_HREDRAW | win32con.CS_VREDRAW
    # wndClass.lpfnWndProc = wndProc
    # wndClass.hInstance = hInstance
    # wndClass.hIcon = win32gui.LoadIcon(0, win32con.IDI_APPLICATION)
    # wndClass.hCursor = win32gui.LoadCursor(0, win32con.IDC_ARROW)
    # wndClass.hbrBackground = win32gui.GetStockObject(win32con.WHITE_BRUSH)
    # wndClass.lpszClassName = szClassName

    hInstance = win32api.GetModuleHandle()

    wndClass = win32gui.WNDCLASS()
    wndClass.hInstance = hInstance
    wndClass.lpszClassName = szClassName
    wndClass.lpfnWndProc = WndProc
    wndClass.style = win32con.CS_DBLCLKS
    # wndClass.cbsize =
    wndClass.hIcon = win32gui.LoadIcon(0, win32con.IDI_APPLICATION)
    # hIconSm
    wndClass.hCursor = win32gui.LoadCursor(0, win32con.IDC_ARROW)
    # wndClass.lpszMenuName = "None"
    # wndClass.cbClsExtra = 0
    wndClass.cbWndExtra = 0
    # wndClass.hbrBackground = int(win32gui.CreateSolidBrush(win32api.RGB(0, 0, 0)))
    wndClass.hbrBackground = win32gui.GetStockObject(win32con.BLACK_BRUSH)

    tWnd = win32gui.FindWindow(0, TargetName)
    if tWnd == 0:
        exit(1)
    tRect = win32gui.GetWindowRect(tWnd)
    #rect = win32gui.GetWindowRect(tWnd)
    wWidth = tRect[2] - tRect[0]
    wHeight = tRect[3] - tRect[1]
    if not tWnd :
        wWidth = 400
        wHeight = 350

    # if not win32gui.RegisterClass(wndClass) :
    #     return 0


        # register window class

    wndClassAtom = None
    try:
        wndClassAtom = win32gui.RegisterClass(wndClass)
    except Exception as e:
        print(e)
        raise e
    hWnd = win32gui.CreateWindowEx(win32con.WS_EX_TOPMOST | win32con.WS_EX_TRANSPARENT, wndClassAtom, "Overlay",
		win32con.WS_POPUP, 100, 90, wWidth, wHeight, 0, 0, hInstance, None)

    # Show & update the window
    win32gui.ShowWindow(hWnd, win32con.SW_SHOW)
    win32gui.UpdateWindow(hWnd)

    win32gui.SetWindowLong(hWnd, win32con.GWL_EXSTYLE, win32gui.GetWindowLong(hWnd, win32con.GWL_EXSTYLE) ^ win32con.WS_EX_LAYERED)
    win32gui.SetLayeredWindowAttributes(hWnd, win32api.RGB(0, 0, 0), 0, win32con.LWA_COLORKEY)

    # win32ui.CreateThread(SetWindow, hWnd)
    t = threading.Thread(target=SetWindow, args=(hWnd,))
    t.daemon = True
    t.start()

#ui
    msg = win32gui.GetMessage(hWnd, 0, 0)
    msg = msg[1]

    # while msg[1]:
    #     win32gui.TranslateMessage(msg)
    #     win32gui.DispatchMessage(msg)
    #     msg = win32gui.GetMessage(hWnd, 0, 0)
    #     msg = msg[1]
    #     print("234345")
    win32gui.PumpMessages()

def WndProc(hWnd, message, wParam, lParam):
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
            print("2346724587295")
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



def SetWindow(hwnd):
    global flag
    while True :
        tWnd = win32gui.FindWindow(0, TargetName)
        print(tWnd)
        if tWnd :
            tRect = win32gui.GetWindowRect(tWnd)
            wWidth = tRect[2] - tRect[0]
            wHeight = tRect[3] - tRect[1]
            dwStyle = win32gui.GetWindowLong(tWnd, win32con.GWL_STYLE)
            if (dwStyle & win32con.WS_BORDER) :
                tRect2 = (tRect[0], tRect[1] + 23, tRect[2], tRect[3])
                wHeight -= 23
            win32gui.MoveWindow(hwnd, tRect2[0], tRect2[1], wWidth, wHeight, True)
        else :
            print("7787")
            flag = 1
            Test()
            exit(1) #ㄴㅇ러ㅏㅜㅎㄹ눓나히ㅏ리;힒ㅎㄴ앓;히ㅏㄹ할나어
            print("12133123")

def Test():
    # while True:
    #     print("Test")
    print("Test")
    exit(1)


if __name__=="__main__":
    WinMain()