import pandas as pd

df = pd.read_csv("remove_outliers_data.csv")
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_row', 100)
print(df.describe())

mean_data = df.mean()[3:]

"""
왼쪽 광대뼈 수직 비율        0.566149
오른쪽 광대뼈 수직 비율       0.565314
왼쪽 눈썹 수직 비율         0.041069
오른쪽 눈썹 수직 비율        0.041686
왼쪽눈 수직 비율           0.065381
오른쪽눈 수직 비율          0.065357
입 수직 비율             0.105573
입 수평 비율             0.398976
왼눈 수평 비율            0.201178
오른눈 수평 비율           0.208523
왼눈 오른쪽 각도          28.860592
왼눈 왼쪽 각도          311.479040
오른눈 오른쪽 각도         45.716758
오른눈 왼쪽 각도         331.592791
입술 왼쪽 각도           28.950460
입술 오른쪽 각도           4.738081
dtype: float64
"""

"""
def classify(new_data):
  score = 1.0

  print(f'코, 왼어깨:{new_data[:, -5]} | 코, 어깨가운데:{new_data[:, -4]} | 코, 오른어깨:{new_data[:, -3]} | 왼팔꿈치 각도:{new_data[:, -2]} | 오른팔꿈치 각도:{new_data[:, -1]}\n')

  if new_data[:, -5] < 0 or new_data[:, -4] < 0 or new_data[:, -3] < 0: # 고개를 푹 숙인경우
    score -= 1

  ######## 어깨 가운데 #########
  if new_data[:, -4] <0.15 and new_data[:, -4] >0.13:
    print('살짝 숙임')
    score -=0.5
  elif new_data[:, -4] <=0.13:
    print('많이 숙임')
    score -= 10.0
  else:
    print('고개 안숙임')
    score += 1.0

  ######## 왼쪽 어깨 #########
  if new_data[:, -5] <0.20 and new_data[:, -5] >0.17:
    print('살짝 기울어짐')
    score -=0.25
  elif new_data[:, -5] <=0.17:
    print('많이 기울어짐')
    score -= 10
  else:
    print('안기울어짐')
    score += 0.5

  ######## 오른쪽 어깨 #########
  if new_data[:, -3] <0.20 and new_data[:, -3] >0.17:
    print('살짝 기울어짐')
    score -=0.25
  elif new_data[:, -3] <=0.17:
    print('많이 기울어짐')
    score -= 10
  else:
    print('안기울어짐')
    score += 0.5


  return score
"""

print(mean_data)
print(mean_data[0])

import cv2
import mediapipe as mp
import math
from math import atan2, degrees
import numpy as np

def point_to_point_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))


def angle_between(p1, p2, p3):  # 3점 사이 각도
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360

    result = deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)

    if result > 180:
        result = 360 - result

    return result

def rate_processing(name,img_point_dict_data):
    rate_name_point_dict = {"왼쪽 광대뼈 수직 비율":[67,50,67,149], "오른쪽 광대뼈 수직 비율":[297,280,297,378], "왼쪽 눈썹 수직 비율":[52,144,67,149], "오른쪽 눈썹 수직 비율":[282,373,297,378],
                       "왼쪽눈 수직 비율":[159,144,67,149], "오른쪽눈 수직 비율":[386,373,297,378], "입 수직 비율":[17,0,10,152], "입 수평 비율":[61,291,288,58], "왼눈 수평 비율":[33,133,356,127],
                       "오른눈 수평 비율":[362,263,356,127]}

    if name in rate_name_point_dict:
        first_rate = point_to_point_distance(img_point_dict_data[rate_name_point_dict[name][0]], img_point_dict_data[rate_name_point_dict[name][1]])
        second_rate = point_to_point_distance(img_point_dict_data[rate_name_point_dict[name][2]], img_point_dict_data[rate_name_point_dict[name][3]])
        result_rate = first_rate/second_rate

        return result_rate

def angle_processing(name,img_point_dict_data):
    angle_name_point_dict = {"왼눈 오른쪽 각도":[159,133,144], "왼눈 왼쪽 각도":[159,33,144], "오른눈 오른쪽 각도":[386,263,373], "오른눈 왼쪽 각도":[386,362,373], "입술 왼쪽 각도":[13,78,14],
                             "입술 오른쪽 각도":[13,308,14]}

    if name in angle_name_point_dict:
        insert_point_list = angle_name_point_dict[name]
        result_angle = angle_between(img_point_dict_data[insert_point_list[0]],img_point_dict_data[insert_point_list[1]],img_point_dict_data[insert_point_list[2]])

        return result_angle



def process_data_rates(img_point_dict_data)->dict:
    POINT = (
    107, 55, 105, 52, 70, 46, 336, 285, 334, 282, 300, 276, 159, 144, 33, 133, 386, 373, 362, 263, 17, 61, 291, 0, 78,
    308, 14, 13,
    10, 297, 389, 356, 288, 378, 152, 149, 58, 127, 162, 67,
    50, 280)  # 10부터 시계방향 50,280(왼쪽 광대, 오른쪽 광대)

    name_list = ["왼쪽 광대뼈 수직 비율","오른쪽 광대뼈 수직 비율","왼쪽 눈썹 수직 비율","오른쪽 눈썹 수직 비율","왼쪽눈 수직 비율","오른쪽눈 수직 비율","입 수직 비율","입 수평 비율","왼눈 수평 비율","오른눈 수평 비율",
                 "왼눈 오른쪽 각도","왼눈 왼쪽 각도","오른눈 오른쪽 각도","오른눈 왼쪽 각도","입술 왼쪽 각도","입술 오른쪽 각도"]

    current_img_processing_data = {}#현재 이미지 값 저장 변수

    for name in name_list:
        if "비율" in name:
            current_img_processing_data[name] = rate_processing(name,img_point_dict_data)
        elif "각도" in name:
            current_img_processing_data[name] = angle_processing(name,img_point_dict_data)
        else:
            print("오류?")

    print(current_img_processing_data)
    return current_img_processing_data

"""**************기준 잡는 함수***************"""
def state_classification(img_porcessing_data, mean_data=mean_data): # img_porcessing_data : 현재 사진에서 각도 비율 계산한것 / mean_data : fer2013 평균 각도 비율 계산한것

    keys_list = list(img_porcessing_data.keys())
    values_gap = {}
    for i, key in zip(range(len(img_porcessing_data)),keys_list):
        values_gap[key] = (img_porcessing_data[keys_list[i]]-mean_data[i])
        # print(f"{key} 차이: {abs(img_porcessing_data[keys_list[i]]-mean_data[i])}")

    print("\n\n두 값 차이",values_gap)
    print(abs(values_gap["왼쪽눈 수직 비율"]))

    # if abs(values_gap["왼쪽눈 수직 비율"]) > 0.009 or abs(values_gap["오른쪽눈 수직 비율"]) > 0.009:
    #     return "졸음 상태"

    if values_gap["왼눈 오른쪽 각도"] > -0.5 or values_gap["왼눈 왼쪽 각도"] > -0.5 or values_gap["오른눈 오른쪽 각도"] > -0.5 or values_gap["오른눈 왼쪽 각도"] > -0.5:
        return "집중"
    else:
        return "졸음상태"

    return "집중"
"""**************기준 잡는 함수***************"""


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
])  ## 얼굴 좌표

# For static images:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:
    file_list = {"boring2.jpg": 1}
    for idx, file in enumerate(file_list):
        # print(file)
        image = cv2.imread(file)
        # Convert the BGR image to RGB before processing.
        image = cv2.resize(image,dsize=(500,500))
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # print("result: {0}".format(results))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            continue
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            # print('face_landmarks:', face_landmarks)
            img_point_data, success_flag = mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

            if success_flag == True:
                print("flag success")
                img_processing_data = process_data_rates(img_point_data)
            else:
                print("flag fail")
                continue

            print("감정결과 출력:",state_classification(img_processing_data))
            cv2.imshow('hi', annotated_image)  ###### 결과 사진 출력
            cv2.waitKey(0)
