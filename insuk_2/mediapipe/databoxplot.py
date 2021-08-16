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
    result = deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)

    if result > 180:
        result = 360 - result

    return result


"""
"""

# sample_data.index = index

# 각 부위의 가로 세로 비율
LEFT_CHEEKBONE_RATE = []
RIGHT_CHEEKBONE_RATE = []
LEFT_EYEBROW_RATE = []
RIGHT_EYEBROW_RATE = []

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
    left_eyebrow_v_rate_result = point_to_point_distance(dataframe.loc[i, "왼눈썹/중간/위/x"],
                                                         dataframe.loc[i, "왼눈썹/중간/위/y"],
                                                         dataframe.loc[i, "왼눈썹/중간/아래/x"],
                                                         dataframe.loc[i, "왼눈썹/중간/아래/y"])  # 왼눈썹 수직 거리

    right_eyebrow_v_rate_result = point_to_point_distance(dataframe.loc[i, "오른눈썹/중간/위/x"],
                                                          dataframe.loc[i, "오른눈썹/중간/위/y"],
                                                          dataframe.loc[i, "오른눈썹/중간/아래/x"],
                                                          dataframe.loc[i, "오른눈썹/중간/아래/y"])  # 오른눈썹 수직 거리

    left_cheekbone_v_rate_result = point_to_point_distance(dataframe.loc[i, "얼굴/11시/x"], dataframe.loc[i, "얼굴/11시/y"],
                                                           dataframe.loc[i, "왼쪽/광대/x"],
                                                           dataframe.loc[i, "왼쪽/광대/y"])  # 머리부터 왼쪽 광대뼈 수직 거리

    right_cheekbone_v_rate_result = point_to_point_distance(dataframe.loc[i, "얼굴/1시/x"], dataframe.loc[i, "얼굴/1시/y"],
                                                            dataframe.loc[i, "오른쪽/광대/x"],
                                                            dataframe.loc[i, "오른쪽/광대/y"])  # 머리부터 오른쪽 광대뼈 수직 거리

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
                                            dataframe.loc[i, ["입술/안쪽/6시/x", "입술/안쪽/6시/x"]])  # 입술 왼쪽 각도

    mouth_right_angle_result = angle_between(dataframe.loc[i, ["입술/안쪽/12시/x", "입술/안쪽/12시/y"]],
                                             dataframe.loc[i, ["입술/안쪽/3시/x", "입술/안쪽/3시/y"]],
                                             dataframe.loc[i, ["입술/안쪽/6시/x", "입술/안쪽/6시/y"]])  # 입술 오른쪽 각도

    LEFT_CHEEKBONE_RATE.append(left_cheekbone_v_rate_result)
    RIGHT_CHEEKBONE_RATE.append(right_cheekbone_v_rate_result)
    LEFT_EYEBROW_RATE.append(left_eyebrow_v_rate_result)
    RIGHT_EYEBROW_RATE.append(right_eyebrow_v_rate_result)
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


left_cheekbone_face_v_result_array = np.array(LEFT_CHEEKBONE_RATE) / np.array(LEFT_FACE_V_RATE)  # 왼쪽 광대뼈 수직 비율
right_cheekbone_face_v_result_array = np.array(RIGHT_CHEEKBONE_RATE) / np.array(RIGHT_FACE_V_RATE)  # 오른쪽 광대뼈 수직 비율
left_eyebrow_face_v_result_array = np.array(LEFT_EYEBROW_RATE) / np.array(LEFT_FACE_V_RATE)  # 왼쪽 눈썹 수직 비율
right_eyebrow_face_v_result_array = np.array(RIGHT_EYEBROW_RATE) / np.array(RIGHT_FACE_V_RATE)  # 오른쪽 눈썹 수직 비율
left_eye_face_v_result_array = np.array(LEFT_EYE_V_RATE) / np.array(LEFT_FACE_V_RATE)  # 왼쪽 눈 수직 비율
right_eye_face_v_result_array = np.array(RIGHT_EYE_V_RATE) / np.array(RIGHT_FACE_V_RATE)  # 오른쪽 눈 수직 비율
middle_mouth_face_v_result_array = np.array(MOUTH_V_RATE) / np.array(MIDDLE_FACE_V_RATE)  # 입 수직 비율
mouth_face_h_result_array = np.array(MOUTH_H_RATE) / np.array(DOWN_FACE_H_RATE)  # 입 수평 비율
left_eye_face_h_result_array = np.array(LEFT_EYE_H_RATE) / np.array(UP_FACE_H_RATE)  # 왼쪽 눈 수평 비율
right_eye_face_h_result_array = np.array(RIGHT_EYE_H_RATE) / np.array(UP_FACE_H_RATE)  # 오른쪽 눈 수평 비율

list_name = [left_cheekbone_face_v_result_array, right_cheekbone_face_v_result_array, left_eyebrow_face_v_result_array,
             right_eyebrow_face_v_result_array,
             left_eye_face_v_result_array, right_eye_face_v_result_array, middle_mouth_face_v_result_array,
             mouth_face_h_result_array, left_eye_face_h_result_array, right_eye_face_h_result_array,
             LEFT_EYE_RIGHT_ANGLE, LEFT_EYE_LEFT_ANGLE, RIGHT_EYE_RIGHT_ANGLE, RIGHT_EYE_LEFT_ANGLE,
             MOUTH_LEFT_ANGLE, MOUTH_RIGHT_ANGLE]

list_string_name = ["왼쪽 광대뼈 수직 비율", "오른쪽 광대뼈 수직 비율", "왼쪽 눈썹 수직 비율", "오른쪽 눈썹 수직 비율",
                    "왼눈 수직 비율", "오른눈 수직 비율", "입술 수직 비율",
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

    # #이상치 중 특정 기준점 이미지만 표시
    # if name in ["왼쪽 광대뼈 수직 비율","오른쪽 광대뼈 수직 비율", "왼쪽 눈썹 수직 비율", "오른쪽 눈썹 수직 비율"]:
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

print("이상값 파일 행 번호", sorted(outliers_file_number))  # 순서의 번호 파일 자체의 번호 아님
print("이상값 파일 개수", len(outliers_file_number))

"""
이상치 데이터 프레임에서 제거하는 부분
"""
#
# print("처음 데이터프레임 개수:", len(dataframe))
# for number_index in sorted(outliers_file_number):
#     dataframe = dataframe.drop(index=number_index)
#
# print(dataframe.index)
# print("박스플롯 이상치 제거한 데이터프레임 개수:", len(dataframe))
#
#
# insuk = [3324, 3325, 3350, 3383, 3400, 3419, 3428, 3447, 3470, 3471, 3474, 3475, 3476, 3489, 3520,
#          3528, 3544, 3577, 3628, 3673, 3684, 3687, 3691, 3692, 3741, 3797, 3819, 3883, 3907, 3911, 4005, 4020, 4052, 4120, 4155, 4157, 4205, 4230, 4263, 4320, 4341, 4372, 4397, 4420, 4502, 4518, 4532, 4601, 4608, 4614, 4720, 4786, 4836, 4859, 4868, 4930, 4933]
#
# sangmin = [1662, 1670, 1678, 1691, 1701, 1715, 1717, 1722, 1729, 1739, 1747, 1771, 1774, 1790, 1808, 1818, 1831, 1868, 1874, 1886, 1921, 1922, 1923, 1974, 1980, 1999, 2043, 2091, 2094, 2097, 2103, 2189, 2228, 2241, 2243, 2256, 2258, 2294, 2303, 2335, 2357, 2381, 2384, 2396, 2473, 2481, 2485, 2487, 2504, 2608, 2720, 2726, 2767, 2808, 2816, 2855, 2857, 2881, 2972, 2977, 2982, 2983, 3007, 3018, 3019, 3020, 3023, 3024, 3099, 3118, 3125, 3163, 3180, 3202, 3214, 3257, 3265]
#
# tekjun = [12, 15, 56, 65, 101, 130, 135, 150, 151, 160, 199, 237, 241, 258, 279, 286, 293, 305, 336, 339, 346, 365, 366, 369, 375, 392, 393, 401, 424, 431, 478, 512, 522, 536, 540, 543, 558, 575, 577, 582, 602, 612, 613, 628, 636, 640, 645, 675, 710, 726, 738, 740, 744, 770, 772, 797, 808, 844, 847, 902, 905, 907, 923, 950, 968, 972, 979, 986, 994, 1000, 1040, 1054, 1056, 1057, 1092, 1100, 1111, 1118, 1128, 1136, 1156, 1160, 1184, 1195, 1207, 1214, 1217, 1218, 1221, 1279, 1288, 1310, 1317, 1318, 1326, 1329, 1337, 1344, 1346, 1347, 1387, 1396, 1400, 1416, 1436, 1448, 1455, 1470, 1491, 1509, 1530, 1571, 1573, 1587, 1593, 1600, 1604, 1612, 1613]
#
# fail_img_list = insuk + sangmin + tekjun
#
# for fail_img_index in fail_img_list:
#     dataframe.drop(dataframe.loc[dataframe['파일번호'] == fail_img_index].index, inplace=True)
#
# print("우리가 거른거 제거한 데이터프레임 개수: ", len(dataframe))

"""
여기까지
"""
# # 이상치 제거된 이미지 표시
# print("-----------------------이상치 제거된 이미지 표시 -------------------")
# print("-----------------------이상치 제거된 이미지 표시 -------------------")
# print("-----------------------이상치 제거된 이미지 표시 -------------------")
#
# pass_list = []
# fail_list = []
# count = int(len(dataframe) / 3 * 2)
# while count < len(dataframe):
#     print(f"현재 행번호: {count} /// q: fial w: pass")
#     if count in outliers_file_number:  # 이상치 세트안에 잇으면 넘기기
#         print(f"{count}행 이상치 다음 번호 이동\n")
#         count = count + 1
#         continue
#     path = "C:\\Users\\insuk\\Desktop\\fer2013\\train\\Neutral"
#     img_path = str(dataframe.loc[count, "파일번호"]) + ".jpg"
#     total_path = path + '/' + img_path
#     print(total_path)
#     image = cv2.imread(total_path, cv2.IMREAD_ANYCOLOR)
#     image = cv2.resize(image, dsize=(200, 200))
#     cv2.imshow("img", image)
#     ret = cv2.waitKey(0)
#     if ret == 119:# w = 119
#         print(f"{count}번 pass\n")
#         pass_list.append(dataframe.loc[count, "파일번호"])
#         count = count + 1
#     elif ret == 113:# q = 113
#         print(f"{count}번 fail\n")
#         fail_list.append(dataframe.loc[count, "파일번호"])
#         count = count + 1
#     else:
#         print("\n#################다시 입력 q: 이상 w: 정상#############")
#
#     cv2.destroyAllWindows()
#
# print("니가 걸러낸거: ",fail_list) # 행번호 아니고 파일 번호임 csv '파일속성'
# print("니가 통과시킨거: ",pass_list)# 행번호 아니고 파일 번호임 csv '파일속성'

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
#     ret = cv2.waitKey(0)
#     if ret == 'w':
#         print("pass")
#         pass_list.append(dataframe.loc[i, "파일번호"])
#     elif ret == 'q':
#         print("fail")
#         fail_list.append(dataframe.loc[i, "파일번호"])
#     else :
#         i = i -1
#
#     cv2.destroyAllWindows()


boxplot_dataframe_3 = {'파일번호': dataframe['파일번호'],
                       '왼쪽 광대뼈 수직 비율': left_cheekbone_face_v_result_array,
                       '오른쪽 광대뼈 수직 비율': right_cheekbone_face_v_result_array,
                       '왼쪽 눈썹 수직 비율': left_eyebrow_face_v_result_array,
                       '오른쪽 눈썹 수직 비율': right_eyebrow_face_v_result_array}

boxplot_dataframe = {'파일번호': dataframe['파일번호'],
                     '왼쪽눈 수직 비율': left_eye_face_v_result_array,
                     '오른쪽눈 수직 비율': right_eye_face_v_result_array,
                     '입 수직 비율': middle_mouth_face_v_result_array,
                     '입 수평 비율': mouth_face_h_result_array,
                     '왼눈 수평 비율': left_eye_face_h_result_array,
                     '오른눈 수평 비율': right_eye_face_h_result_array}

boxplot_dataframe2 = {'파일번호': dataframe['파일번호'],
                      "왼눈 오른쪽 각도": LEFT_EYE_RIGHT_ANGLE,
                      "왼눈 왼쪽 각도": LEFT_EYE_LEFT_ANGLE,
                      "오른눈 오른쪽 각도": RIGHT_EYE_RIGHT_ANGLE,
                      "오른눈 왼쪽 각도": RIGHT_EYE_LEFT_ANGLE,
                      "입술 왼쪽 각도": MOUTH_LEFT_ANGLE,
                      "입술 오른쪽 각도": MOUTH_RIGHT_ANGLE}

#거를거 저장할 데이터 프레임
save_data_dataframe = {'파일번호': dataframe['파일번호'],
                       '왼쪽 광대뼈 수직 비율': left_cheekbone_face_v_result_array,
                       '오른쪽 광대뼈 수직 비율': right_cheekbone_face_v_result_array,
                       '왼쪽 눈썹 수직 비율': left_eyebrow_face_v_result_array,
                       '오른쪽 눈썹 수직 비율': right_eyebrow_face_v_result_array,
                       '왼쪽눈 수직 비율': left_eye_face_v_result_array,
                       '오른쪽눈 수직 비율': right_eye_face_v_result_array,
                       '입 수직 비율': middle_mouth_face_v_result_array,
                       '입 수평 비율': mouth_face_h_result_array,
                       '왼눈 수평 비율': left_eye_face_h_result_array,
                       '오른눈 수평 비율': right_eye_face_h_result_array,
                       "왼눈 오른쪽 각도": LEFT_EYE_RIGHT_ANGLE,
                       "왼눈 왼쪽 각도": LEFT_EYE_LEFT_ANGLE,
                       "오른눈 오른쪽 각도": RIGHT_EYE_RIGHT_ANGLE,
                       "오른눈 왼쪽 각도": RIGHT_EYE_LEFT_ANGLE,
                       "입술 왼쪽 각도": MOUTH_LEFT_ANGLE,
                       "입술 오른쪽 각도": MOUTH_RIGHT_ANGLE
                       }

boxplot_dataframe = pd.DataFrame(boxplot_dataframe2)
save_dataframe = pd.DataFrame(save_data_dataframe)#거를거 저장할 데이터 프레임

# ############################################삭제할부분#############
# plt.figure(figsize=(30, 30))
# boxplot = boxplot_dataframe.boxplot(fontsize=10, rot=30)
# plt.show()
# ###################################################################
""""""  # 최종 이상치 제거한 boxplot 그리는 부분

print("처음 데이터프레임 개수:", len(boxplot_dataframe))
for number_index in sorted(outliers_file_number):
    boxplot_dataframe = boxplot_dataframe.drop(index=number_index)
    save_dataframe = save_dataframe.drop(index=number_index)#거를거 저장할 데이터 프레임

print(boxplot_dataframe.index)
print("박스플롯 이상치 제거한 데이터프레임 개수:", len(boxplot_dataframe))

insuk = [3324, 3325, 3350, 3383, 3400, 3419, 3428, 3447, 3470, 3471, 3474, 3475, 3476, 3489, 3520,
         3528, 3544, 3577, 3628, 3673, 3684, 3687, 3691, 3692, 3741, 3797, 3819, 3883, 3907, 3911, 4005, 4020, 4052,
         4120, 4155, 4157, 4205, 4230, 4263, 4320, 4341, 4372, 4397, 4420, 4502, 4518, 4532, 4601, 4608, 4614, 4720,
         4786, 4836, 4859, 4868, 4930, 4933]

sangmin = [1662, 1670, 1678, 1691, 1701, 1715, 1717, 1722, 1729, 1739, 1747, 1771, 1774, 1790, 1808, 1818, 1831, 1868,
           1874, 1886, 1921, 1922, 1923, 1974, 1980, 1999, 2043, 2091, 2094, 2097, 2103, 2189, 2228, 2241, 2243, 2256,
           2258, 2294, 2303, 2335, 2357, 2381, 2384, 2396, 2473, 2481, 2485, 2487, 2504, 2608, 2720, 2726, 2767, 2808,
           2816, 2855, 2857, 2881, 2972, 2977, 2982, 2983, 3007, 3018, 3019, 3020, 3023, 3024, 3099, 3118, 3125, 3163,
           3180, 3202, 3214, 3257, 3265]

tekjun = [12, 15, 56, 65, 101, 130, 135, 150, 151, 160, 199, 237, 241, 258, 279, 286, 293, 305, 336, 339, 346, 365, 366,
          369, 375, 392, 393, 401, 424, 431, 478, 512, 522, 536, 540, 543, 558, 575, 577, 582, 602, 612, 613, 628, 636,
          640, 645, 675, 710, 726, 738, 740, 744, 770, 772, 797, 808, 844, 847, 902, 905, 907, 923, 950, 968, 972, 979,
          986, 994, 1000, 1040, 1054, 1056, 1057, 1092, 1100, 1111, 1118, 1128, 1136, 1156, 1160, 1184, 1195, 1207,
          1214, 1217, 1218, 1221, 1279, 1288, 1310, 1317, 1318, 1326, 1329, 1337, 1344, 1346, 1347, 1387, 1396, 1400,
          1416, 1436, 1448, 1455, 1470, 1491, 1509, 1530, 1571, 1573, 1587, 1593, 1600, 1604, 1612, 1613]

fail_img_list = insuk + sangmin + tekjun

for fail_img_index in fail_img_list:
    boxplot_dataframe.drop(boxplot_dataframe.loc[boxplot_dataframe['파일번호'] == fail_img_index].index, inplace=True)
    save_dataframe.drop(save_dataframe.loc[save_dataframe['파일번호'] == fail_img_index].index, inplace=True)

print("우리가 거른거 제거한 데이터프레임 개수: ", len(boxplot_dataframe))

del boxplot_dataframe['파일번호']  # 파일번호 삭제

#이상치 제거된 데이터프레임 저장
save_dataframe.reset_index(inplace=True)
save_dataframe.to_csv(f"remove_outliers_data.csv",encoding='utf-8-sig', mode='w') # 저장 하는부분

""""""

plt.figure(figsize=(30, 30))
boxplot = boxplot_dataframe.boxplot(fontsize=10, rot=30)
plt.show()
