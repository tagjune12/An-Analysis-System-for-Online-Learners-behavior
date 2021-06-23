from PIL import ImageGrab
import numpy as np

class CaptureBoard:
    def __init__(self):
        self.capture_size = (0, 0, 1920, 1080)

    def captureScreen(self):
        image = ImageGrab.grab(bbox=self.capture_size)
        image = np.array(image)
        return image