from PIL import ImageGrab
import numpy as np

# from abc import ABCMeta, abstractmethod

# class CaptureBoard: # Obsever
#     __metaclass__ = ABCMeta
#
#     @abstractmethod
#     def update(self, size:tuple):
#         pass
#
#     @abstractmethod
#     def register_subject(self):
#         pass
#
# class ScreenCaptureBoard(CaptureBoard): # Concrete Obsever
#     def __init__(self):
#         self.capture_size = (0, 0, 1920, 1080)
#
#     def update(self, size:tuple):
#         self.capture_size = size
#
#     def captureScreen(self):
#         image = ImageGrab.grab(bbox=self.capture_size)
#         image = np.array(image)
#         return image

# class CaptureBoard:
#     def __init__(self):
#         self.capture_size = (0, 0, 1920, 1080)
#
#     def captureScreen(self):
#         image = ImageGrab.grab(bbox=self.capture_size)
#         image = np.array(image)
#         return image
#
#     def set_capture_size(self, window_size):
#         self.capture_size = window_size

# 싱글톤 패턴
class CaptureBoard(object):
    def __new__(cls):
        if not hasattr(cls,'instance'):
            # print('CaptureBoard is created')
            cls.instance =super(CaptureBoard,cls).__new__(cls)
            cls.capture_size = [0, 0, 1920, 1080]
        else:
            # print('recycle')
            return cls.instance


    def captureScreen(self):
        image = ImageGrab.grab(bbox=self.capture_size)
        image = np.array(image)
        return image

    def set_capture_size(self, window_size):
        self.capture_size = window_size

    def get_capture_size(self):
        return self.capture_size