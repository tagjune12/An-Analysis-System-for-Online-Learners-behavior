import math

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataframe = pd.read_csv("face_point_data.csv")
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_row', 100)

from matplotlib import font_manager, rc

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)


def outliers_data_remove(data):
    q1, q3 = np.percentile(data, [25, 75])  # 데이터의 25지점 75지점의 값 알아내기
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)  # 경계선 위 값
    upper_bound = q3 + (iqr * 1.5)  # 경계선 아래 값

    return np.where((data > upper_bound) | (data < lower_bound))  # where: index


def point_to_point_distance(x1, y1, x2, y2):
    return math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))


from math import atan2, degrees


def angle_between(p1, p2, p3):  # 3점 사이 각도
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360
    return deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)


"""
"""
# sample_data.index = index

# 각 부위의 가로 세로 비율
LEFT_EYE_V_RATE = []
LEFT_FACE_V_RATE = []
RIGHT_EYE_V_RATE = []
RIGHT_FACE_V_RATE = []
MOUTH_V_RATE = []
MIDDLE_FACE_V_RATE = []
MOUTH_H_RATE = []
DOWN_FACE_H_RATE = []
UP_FACE_H_RATE = []
LEFT_EYE_H_RATE = []
RIGHT_EYE_H_RATE = []

# 각 부위의 각도
LEFT_EYE_LEFT_ANGLE = []
LEFT_EYE_RIGHT_ANGLE = []
RIGHT_EYE_LEFT_ANGLE = []
RIGHT_EYE_RIGHT_ANGLE = []
MOUTH_LEFT_ANGLE = []
MOUTH_RIGHT_ANGLE = []

for i in range(0, len(dataframe)):
    left_eye_v_rate_result = point_to_point_distance(dataframe.loc[i, "왼눈/12시/x"], dataframe.loc[i, "왼눈/12시/y"],
                                                     dataframe.loc[i, "왼눈/6시/x"],
                                                     dataframe.loc[i, "왼눈/6시/y"])  # 왼눈 수직 거리

    left_face_v_rate_result = point_to_point_distance(dataframe.loc[i, "얼굴/11시/x"], dataframe.loc[i, "얼굴/11시/y"],
                                                      dataframe.loc[i, "얼굴/7시/x"],
                                                      dataframe.loc[i, "얼굴/7시/y"])  # 왼쪽 얼굴 수직 거리

    right_eye_v_rate_result = point_to_point_distance(dataframe.loc[i, "오른눈/12시/x"], dataframe.loc[i, "오른눈/12시/y"],
                                                      dataframe.loc[i, "오른눈/6시/x"],
                                                      dataframe.loc[i, "오른눈/6시/y"])  # 오른눈 수직 거리

    right_face_v_rate_result = point_to_point_distance(dataframe.loc[i, "얼굴/1시/x"], dataframe.loc[i, "얼굴/1시/y"],
                                                       dataframe.loc[i, "얼굴/5시/x"],
                                                       dataframe.loc[i, "얼굴/5시/y"])  # 오른쪽 얼굴 수직 거리

    mouth_v_rate_result = point_to_point_distance(dataframe.loc[i, "입술/바깥/12시/x"], dataframe.loc[i, "입술/바깥/12시/y"],
                                                  dataframe.loc[i, "입술/바깥/6시/x"],
                                                  dataframe.loc[i, "입술/바깥/6시/y"])  # 입술 수직 거리

    middle_face_v_rate_result = point_to_point_distance(dataframe.loc[i, "얼굴/12시/x"], dataframe.loc[i, "얼굴/12시/y"],
                                                        dataframe.loc[i, "얼굴/6시/x"],
                                                        dataframe.loc[i, "얼굴/6시/y"])  # 가운데 얼굴 수직 거리

    down_face_h_rate_result = point_to_point_distance(dataframe.loc[i, "얼굴/4시/x"], dataframe.loc[i, "얼굴/4시/y"],
                                                      dataframe.loc[i, "얼굴/8시/x"],
                                                      dataframe.loc[i, "얼굴/8시/y"])  # 얼굴 아래 수평 거리

    mouth_h_rate_result = point_to_point_distance(dataframe.loc[i, "입술/바깥/9시/x"], dataframe.loc[i, "입술/바깥/9시/y"],
                                                  dataframe.loc[i, "입술/바깥/3시/x"],
                                                  dataframe.loc[i, "입술/바깥/3시/y"])  # 입 수평 거리

    left_eye_h_rate_result = point_to_point_distance(dataframe.loc[i, "왼눈/9시/x"], dataframe.loc[i, "왼눈/9시/y"],
                                                     dataframe.loc[i, "왼눈/3시/x"],
                                                     dataframe.loc[i, "왼눈/3시/y"])  # 왼눈 수평 거리

    right_eye_h_rate_result = point_to_point_distance(dataframe.loc[i, "오른눈/9시/x"], dataframe.loc[i, "오른눈/9시/y"],
                                                      dataframe.loc[i, "오른눈/3시/x"],
                                                      dataframe.loc[i, "오른눈/3시/y"])  # 오른눈 수평 거리

    up_face_h_rate_result = point_to_point_distance(dataframe.loc[i, "얼굴/9시/x"], dataframe.loc[i, "얼굴/9시/y"],
                                                    dataframe.loc[i, "얼굴/3시/x"],
                                                    dataframe.loc[i, "얼굴/3시/y"])  # 얼굴 수평 거리

    left_eye_right_angle_result = angle_between(dataframe.loc[i, ["왼눈/12시/x", "왼눈/12시/y"]],
                                                dataframe.loc[i, ["왼눈/3시/x", "왼눈/3시/y"]],
                                                dataframe.loc[i, ["왼눈/6시/x", "왼눈/6시/y"]])  # 왼눈 오른쪽 각도

    left_eye_left_angle_result = angle_between(dataframe.loc[i, ["왼눈/12시/x", "왼눈/12시/y"]],
                                               dataframe.loc[i, ["왼눈/9시/x", "왼눈/9시/y"]],
                                               dataframe.loc[i, ["왼눈/6시/x", "왼눈/6시/y"]])  # 왼눈 왼쪽 각도

    right_eye_right_angle_result = angle_between(dataframe.loc[i, ["오른눈/12시/x", "오른눈/12시/y"]],
                                                 dataframe.loc[i, ["오른눈/3시/x", "오른눈/3시/y"]],
                                                 dataframe.loc[i, ["오른눈/6시/x", "오른눈/6시/y"]])  # 오른눈 오른쪽 각도

    right_eye_left_angle_result = angle_between(dataframe.loc[i, ["오른눈/12시/x", "오른눈/12시/y"]],
                                                dataframe.loc[i, ["오른눈/9시/x", "오른눈/9시/y"]],
                                                dataframe.loc[i, ["오른눈/6시/x", "오른눈/6시/y"]])  # 오른눈 왼쪽 각도

    mouth_left_angle_result = angle_between(dataframe.loc[i, ["입술/안쪽/12시/x", "입술/안쪽/12시/y"]],
                                            dataframe.loc[i, ["입술/안쪽/9시/x", "입술/안쪽/9시/x"]],
                                            dataframe.loc[i, ["입술/안쪽/6시/x", "입술/안쪽/6시/x"]])  # 오른눈 왼쪽 각도

    mouth_right_angle_result = angle_between(dataframe.loc[i, ["입술/안쪽/12시/x", "입술/안쪽/12시/y"]],
                                             dataframe.loc[i, ["입술/안쪽/3시/x", "입술/안쪽/3시/y"]],
                                             dataframe.loc[i, ["입술/안쪽/6시/x", "입술/안쪽/6시/y"]])  # 오른눈 왼쪽 각도

    LEFT_EYE_V_RATE.append(left_eye_v_rate_result)
    LEFT_FACE_V_RATE.append(left_face_v_rate_result)
    RIGHT_EYE_V_RATE.append(right_eye_v_rate_result)
    RIGHT_FACE_V_RATE.append(right_face_v_rate_result)
    MOUTH_V_RATE.append(mouth_v_rate_result)
    MOUTH_H_RATE.append(mouth_h_rate_result)
    DOWN_FACE_H_RATE.append(down_face_h_rate_result)
    MIDDLE_FACE_V_RATE.append(middle_face_v_rate_result)
    UP_FACE_H_RATE.append(up_face_h_rate_result)
    LEFT_EYE_H_RATE.append(left_eye_h_rate_result)
    RIGHT_EYE_H_RATE.append(right_eye_h_rate_result)

    LEFT_EYE_RIGHT_ANGLE.append(left_eye_right_angle_result)
    LEFT_EYE_LEFT_ANGLE.append(left_eye_left_angle_result)
    RIGHT_EYE_RIGHT_ANGLE.append(right_eye_right_angle_result)
    RIGHT_EYE_LEFT_ANGLE.append(right_eye_left_angle_result)
    MOUTH_LEFT_ANGLE.append(mouth_left_angle_result)
    MOUTH_RIGHT_ANGLE.append(mouth_right_angle_result)

left_eye_face_v_result_array = np.array(LEFT_EYE_V_RATE) / np.array(LEFT_FACE_V_RATE)  # 왼쪽 눈 수직 비율
right_eye_face_v_result_array = np.array(RIGHT_EYE_V_RATE) / np.array(RIGHT_FACE_V_RATE)  # 오른쪽 눈 수직 비율
middle_mouth_face_v_result_array = np.array(MOUTH_V_RATE) / np.array(MIDDLE_FACE_V_RATE)  # 입 수직 비율
mouth_face_h_result_array = np.array(MOUTH_H_RATE) / np.array(DOWN_FACE_H_RATE)  # 입 수평 비율
left_eye_face_h_result_array = np.array(LEFT_EYE_H_RATE) / np.array(UP_FACE_H_RATE)  # 왼쪽 눈 수평 비율
right_eye_face_h_result_array = np.array(RIGHT_EYE_H_RATE) / np.array(UP_FACE_H_RATE)  # 오른쪽 눈 수평 비율

list_name = [left_eye_face_v_result_array, right_eye_face_v_result_array, middle_mouth_face_v_result_array,
             mouth_face_h_result_array, left_eye_face_h_result_array, right_eye_face_h_result_array,
             LEFT_EYE_RIGHT_ANGLE, LEFT_EYE_LEFT_ANGLE, RIGHT_EYE_RIGHT_ANGLE, RIGHT_EYE_LEFT_ANGLE,
             MOUTH_LEFT_ANGLE, MOUTH_RIGHT_ANGLE]

list_string_name = ["왼눈 수직 비율", "오른눈 수직 비율", "입술 수직 비율",
                    "입술 수평 비율", "왼눈 수평 비율", "오른눈 수평 비율",
                    "왼눈 오른쪽 각도", "왼눈 왼쪽 각도", "오른눈 오른쪽 각도", "오른눈 왼쪽 각도",
                    "입술 왼쪽 각도", "입술 오른쪽 각도"]

outliers_file_number = set()  # 이상치

for name, i in zip(list_string_name, list_name):
    result_dataframe = pd.DataFrame(i)
    outliers_result = outliers_data_remove(result_dataframe)
    print(name, outliers_result[0])
    print(result_dataframe.loc[outliers_result[0]])
    print(name, len(outliers_result[0]))
    outliers_file_number.update(list(outliers_result[0]))


    ##이상치 이미지 표시
    # for j in outliers_result[0]:
    #     path = "C:\\Users\\insuk\\Desktop\\fer2013\\train\\Neutral"
    #     img_path = str(dataframe.loc[j, "파일번호"]) + ".jpg"  # 그 행에 해당하는 파일번호 사진
    #     total_path = path + '/' + img_path
    #     print(total_path)
    #     image = cv2.imread(total_path, cv2.IMREAD_ANYCOLOR)
    #     image = cv2.resize(image, dsize=(200, 200))
    #     cv2.imshow("img", image)
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()

    ##이상치 중 특정 기준점 이미지만 표시
    # if name == "입술 왼쪽 각도":
    #     # 이미지 표시
    #     for j in outliers_result[0]:
    #         path = "C:\\Users\\insuk\\Desktop\\fer2013\\train\\Neutral"
    #         img_path = str(dataframe.loc[j, "파일번호"]) + ".jpg" # 그 행에 해당하는 파일번호 사진
    #         total_path = path + '/' + img_path
    #         print(total_path)
    #         image = cv2.imread(total_path, cv2.IMREAD_ANYCOLOR)
    #         image = cv2.resize(image, dsize=(200, 200))
    #         cv2.imshow("img", image)
    #         cv2.waitKey()
    #         cv2.destroyAllWindows()

print("이상값 파일 행 번호", sorted(outliers_file_number))# 순서의 번호 파일 자체의 번호 아님
print("이상값 파일 개수", len(outliers_file_number))


# 이상치 제거된 이미지 표시
# print("-----------------------이상치 제거된 이미지 표시 -------------------")
# print("-----------------------이상치 제거된 이미지 표시 -------------------")
# print("-----------------------이상치 제거된 이미지 표시 -------------------")
#
# for i in range(0, len(dataframe)):
#     print("현재 행번호: ",i)
#     if i in outliers_file_number:  # 이상치 세트안에 잇으면 넘기기
#         continue
#     path = "C:\\Users\\insuk\\Desktop\\fer2013\\train\\Neutral"
#     img_path = str(dataframe.loc[i, "파일번호"]) + ".jpg"
#     total_path = path + '/' + img_path
#     print(total_path)
#     image = cv2.imread(total_path, cv2.IMREAD_ANYCOLOR)
#     image = cv2.resize(image, dsize=(200, 200))
#     cv2.imshow("img", image)
#     cv2.waitKey()
#     cv2.destroyAllWindows()

boxplot_dataframe = {'왼쪽눈 수직 비율': left_eye_face_v_result_array,
                     '오른쪽눈 수직 비율': right_eye_face_v_result_array,
                     '입 수직 비율': middle_mouth_face_v_result_array,
                     '입 수평 비율': mouth_face_h_result_array,
                     '왼눈 수평 비율': left_eye_face_h_result_array,
                     '오른눈 수평 비율': right_eye_face_h_result_array}

boxplot_dataframe2 = {"왼눈 오른쪽 각도": LEFT_EYE_RIGHT_ANGLE,
                      "왼눈 왼쪽 각도": LEFT_EYE_LEFT_ANGLE,
                      "오른눈 오른쪽 각도": RIGHT_EYE_RIGHT_ANGLE,
                      "오른눈 왼쪽 각도": RIGHT_EYE_LEFT_ANGLE,
                      "입술 왼쪽 각도": MOUTH_LEFT_ANGLE,
                      "입술 오른쪽 각도": MOUTH_RIGHT_ANGLE}

boxplot_dataframe2 = pd.DataFrame(boxplot_dataframe2)
plt.figure(figsize=(30, 30))
boxplot = boxplot_dataframe2.boxplot(fontsize=10, rot=30)
plt.show()
