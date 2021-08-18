from PIL import ImageGrab
import numpy as np

from abc import ABCMeta, abstractmethod

class CaptureBoard: # Obsever
    __metaclass__ = ABCMeta

    @abstractmethod
    def update(self, size:tuple):
        pass

    @abstractmethod
    def register_subject(self):
        pass

class ScreenCaptureBoard(CaptureBoard): # Concrete Obsever
    def __init__(self):
        self.capture_size = (0, 0, 1920, 1080)

    def update(self, size:tuple):
        self.capture_size = size

    def captureScreen(self):
        image = ImageGrab.grab(bbox=self.capture_size)
        image = np.array(image)
        return image

# class CaptureBoard:
#     def __init__(self):
#         self.capture_size = (0, 0, 1920, 1080)
#
#     def captureScreen(self):
#         image = ImageGrab.grab(bbox=self.capture_size)
#         image = np.array(image)
#         return image