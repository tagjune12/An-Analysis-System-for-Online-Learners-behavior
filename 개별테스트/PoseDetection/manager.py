import screencapture
import detect_person
import detect_feature
# import classification_module
import display
from Overlay import overlay_2

class Manager():
    def __new__(cls):
        if not hasattr(cls,'instance'):
            print('Manager is created  1')
            cls.instance = super(Manager,cls).__new__(cls)
            cls.captureboard = screencapture.CaptureBoard()
            cls.peopledetector = detect_person.PeopleDetector()
            cls.featuredetector = detect_feature.FeatureDetector()
            cls.display = display.Display()
            cls.overlay = overlay_2.Sticker('red.gif', xy=[300, 300], size=0.3, on_top=True)

        else:
            print('recycle')


        return cls.instance


    def TossWindowSize(self):
        self.captureboard.set_capture_size(self.overlay.window_size)