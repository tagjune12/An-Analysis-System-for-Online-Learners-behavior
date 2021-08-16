import cv2
import keyboard
import mouse
import mediapipe as mp
import numpy as np
from PIL import ImageGrab
import time         ####
import numpy as np  ####
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

FACE_CONNECTIONS = frozenset([
    # # Lips.
    (61, 146),
    (146, 91),
    (91, 181),
    (181, 84),
    (84, 17),
    (17, 314),
    (314, 405),
    (405, 321),
    (321, 375),
    (375, 291),
    (61, 185),
    (185, 40),
    (40, 39),
    (39, 37),
    (37, 0),
    (0, 267),
    (267, 269),
    (269, 270),
    (270, 409),
    (409, 291),
    (78, 95),
    (95, 88),
    (88, 178),
    (178, 87),
    (87, 14),
    (14, 317),
    (317, 402),
    (402, 318),
    (318, 324),
    (324, 308),
    (78, 191),
    (191, 80),
    (80, 81),
    (81, 82),
    (82, 13),
    (13, 312),
    (312, 311),
    (311, 310),
    (310, 415),
    (415, 308),
    # Left eye.
    (263, 249),
    (249, 390),
    (390, 373),
    (373, 374),
    (374, 380),
    (380, 381),
    (381, 382),
    (382, 362),
    (263, 466),
    (466, 388),
    (388, 387),
    (387, 386),
    (386, 385),
    (385, 384),
    (384, 398),
    (398, 362),
    # Left eyebrow.
    (276, 283),
    (283, 282),
    (282, 295),
    (295, 285),
    (300, 293),
    (293, 334),
    (334, 296),
    (296, 336),
    # Right eye.
    (33, 7),
    (7, 163),
    (163, 144),
    (144, 145),
    (145, 153),
    (153, 154),
    (154, 155),
    (155, 133),
    (33, 246),
    (246, 161),
    (161, 160),
    (160, 159),
    (159, 158),
    (158, 157),
    (157, 173),
    (173, 133),
    # Right eyebrow.
    (46, 53),
    (53, 52),
    (52, 65),
    (65, 55),
    (70, 63),
    (63, 105),
    (105, 66),
    (66, 107),
    # Face oval.
    (10, 338),
    (338, 297),
    (297, 332),
    (332, 284),
    (284, 251),
    (251, 389),
    (389, 356),
    (356, 454),
    (454, 323),
    (323, 361),
    (361, 288),
    (288, 397),
    (397, 365),
    (365, 379),
    (379, 378),
    (378, 400),
    (400, 377),
    (377, 152),
    (152, 148),
    (148, 176),
    (176, 149),
    (149, 150),
    (150, 136),
    (136, 172),
    (172, 58),
    (58, 132),
    (132, 93),
    (93, 234),
    (234, 127),
    (127, 162),
    (162, 21),
    (21, 54),
    (54, 103),
    (103, 67),
    (67, 109),
    (109, 10)
])## 얼굴 좌표

# For static images:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5) as face_mesh:
  file_list = {"asd123.jfif":1 }
  for idx, file in enumerate(file_list):
    #print(file)
    image = cv2.imread(file)
    # Convert the BGR image to RGB before processing.
    # image = cv2.resize(image,dsize=(300,300))
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    print("result: {0}".format(results))

    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()
    for face_landmarks in results.multi_face_landmarks:
      print('face_landmarks:', face_landmarks)
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACE_CONNECTIONS,
          landmark_drawing_spec=drawing_spec,
          connection_drawing_spec=drawing_spec)
      cv2.imshow('hi', annotated_image)###### 결과 사진 출력
      cv2.waitKey(0)
      # time.sleep(3)
    print('face_landmarks:', face_landmarks)
    # cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    # for i in FACE_CONNECTIONS:
    #     annotated_image = cv2.circle(annotated_image, idx_to_coordinates.values(), 1, (0, 0, 255), -1)
    #     cv2.imshow('hi',annotated_image)
    #     cv2.waitKey(0)
####
# def set_roi():
#     global ROI_SET, x1, y1, x2, y2
#     ROI_SET = False
#     print("Select your ROI using mouse drag.")
#     while(mouse.is_pressed() == False):
#         x1, y1 = mouse.get_position()
#         while(mouse.is_pressed() == True):
#             x2, y2 = mouse.get_position()
#             while(mouse.is_pressed() == False):
#                 print("Your ROI : {0}, {1}, {2}, {3}".format(x1, y1, x2, y2))
#                 ROI_SET = True
#                 return
# keyboard.add_hotkey("ctrl+1", lambda: set_roi())
# ROI_SET = False
# x1, y1, x2, y2 = 0, 0, 0, 0
# while True:
#     if ROI_SET == True:
#         image = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(x1, y1, x2, y2))), cv2.COLOR_BGR2RGB)
#         cv2.imshow("image", image)
#         key = cv2.waitKey(100)
#         if key == ord("q"):
#             print("Quit")
#             break
# cv2.destroyAllWindows()
####비디오 검출부분
# # For webcam input:
# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# #cap = cv2.VideoCapture(0)
# # cap = cv2.VideoCapture("oking.mp4")
# with mp_face_mesh.FaceMesh(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5,
#     max_num_faces=2) as face_mesh:
#   # while cap.isOpened():
#   while True:
#     image = ImageGrab.grab(bbox=(50, 200, 1000, 1000))  # 스크린을 캡쳐하여 변수에 저장##@!#!@
#     image = np.array(image)  # 이미지를 배열로 변환##@!#!@#!@
#     # success, image = cap.read()
#     # if not success:
#     #   print("Ignoring empty camera frame.")
#     #   # If loading a video, use 'break' instead of 'continue'.
#     #   break
#
#     # Flip the image horizontally for a later selfie-view display, and convert
#     # the BGR image to RGB.
#     # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     results = face_mesh.process(image)
#
#     # Draw the face mesh annotations on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     if results.multi_face_landmarks:
#       for face_landmarks in results.multi_face_landmarks:
#         mp_drawing.draw_landmarks(
#             image=image,
#             landmark_list=face_landmarks,
#             connections=mp_face_mesh.FACE_CONNECTIONS,
#             landmark_drawing_spec=drawing_spec,
#             connection_drawing_spec=drawing_spec)
#     # image = cv2.resize(image,dsize=(1000, 1000))
#     cv2.imshow('MediaPipe FaceMesh', image)
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# # cap.release()
