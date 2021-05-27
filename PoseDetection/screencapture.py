from PIL import ImageGrab
import numpy as np

# image = np.array()

class CaptureBoard:

    def captureScreen(self):
        # global image
        image = ImageGrab.grab(bbox=(0, 0, 1920, 1080))
        image = np.array(image)
        return image

    # def get_frame(self):
    #     global image
    #     return image