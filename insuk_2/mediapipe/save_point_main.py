import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import drawing_utils_copy
import os
import natsort

mp_drawing = drawing_utils_copy
mp_face_mesh = mp.solutions.face_mesh

# POINT = (107,55,105,52,70,46,336,285,334,282,300,276,159,144,33,133,386,373,362,263,17,61,291,0,78,308,14,13)

file_list = os.listdir(r"C:\Users\insuk\Desktop\fer2013\train\Neutral")# fer2013 데이터셋 위치
file_list = np.array(file_list)
file_list = natsort.natsorted(file_list) # 정렬
file_list_dataframe  = pd.DataFrame(file_list)



point_data_list = []

print(file_list_dataframe)
for idx, i in enumerate(file_list_dataframe.values) :
    if idx == 50: break
    print(i)

# For static images:
IMAGE_FILES = {"iu1.jfif":1 }
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5) as face_mesh:
  for idx, file in enumerate(file_list):
    if idx%100==0:
        print(idx)
    image = cv2.imread("C:\\Users\\insuk\\Desktop\\fer2013\\train\\Neutral\\"+file)
    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) ##원래 변환
    # results = face_mesh.process(cv2.cvtColor(image,cv2.COLOR_GRAY2RGB))

    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()
    for face_landmarks in results.multi_face_landmarks:
      # print('face_landmarks:', face_landmarks)
      face_point_index, success_flag = mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACE_CONNECTIONS,
          landmark_drawing_spec=drawing_spec,
          connection_drawing_spec=drawing_spec,
          file_name = file)

      if success_flag == True:
          point_data_list.append(face_point_index)
      else:
          continue

      # # print(type(face_point_index))
      # if 'point_dataframe' not in locals():
      #     point_dataframe = pd.DataFrame(face_point_index)
      # else:
      #     point_dataframe.append(face_point_index)

  # print(point_data_list)
  point_data_list = np.array(point_data_list)
  print(point_data_list[0])
  print(np.shape(point_data_list))
  point_data_list = np.reshape(point_data_list, (-1,85))
  point_dataframe = pd.DataFrame(point_data_list,columns=["파일번호","왼눈썹/안쪽/위/x","왼눈썹/안쪽/위/y","왼눈썹/안쪽/아래/x","왼눈썹/안쪽/아래/y","왼눈썹/중간/위/x","왼눈썹/중간/위/y","왼눈썹/중간/아래/x","왼눈썹/중간/아래/y","왼눈썹/바깥/위/x","왼눈썹/바깥/위/y","왼눈썹/바깥/아래/x","왼눈썹/바깥/아래/y",
                                                          "오른눈썹/안쪽/위/x","오른눈썹/안쪽/위/y","오른눈썹/안쪽/아래/x","오른눈썹/안쪽/아래/y","오른눈썹/중간/위/x","오른눈썹/중간/위/y","오른눈썹/중간/아래/x","오른눈썹/중간/아래/y","오른눈썹/바깥/위/x","오른눈썹/바깥/위/y","오른눈썹/바깥/아래/x","오른눈썹/바깥/아래/y",
                                                          "왼눈/12시/x","왼눈/12시/y","왼눈/6시/x","왼눈/6시/y","왼눈/9시/x","왼눈/9시/y","왼눈/3시/x","왼눈/3시/y",
                                                          "오른눈/12시/x","오른눈/12시/y","오른눈/6시/x","오른눈/6시/y","오른눈/9시/x","오른눈/9시/y","오른눈/3시/x","오른눈/3시/y",
                                                          "입술/바깥/6시/x","입술/바깥/6시/y","입술/바깥/9시/x","입술/바깥/9시/y","입술/바깥/3시/x","입술/바깥/3시/y","입술/바깥/12시/x","입술/바깥/12시/y",
                                                          "입술/안쪽/9시/x","입술/안쪽/9시/y","입술/안쪽/3시/x","입술/안쪽/3시/y","입술/안쪽/6시/x","입술/안쪽/6시/y","입술/안쪽/12시/x","입술/안쪽/12시/y",
                                                          "얼굴/12시/x","얼굴/12시/y","얼굴/1시/x","얼굴/1시/y","얼굴/2시/x","얼굴/2시/y","얼굴/3시/x","얼굴/3시/y","얼굴/4시/x","얼굴/4시/y","얼굴/5시/x","얼굴/5시/y",
                                                          "얼굴/6시/x","얼굴/6시/y","얼굴/7시/x","얼굴/7시/y","얼굴/8시/x","얼굴/8시/y","얼굴/9시/x","얼굴/9시/y","얼굴/10시/x","얼굴/10시/y","얼굴/11시/x","얼굴/11시/y",
                                                          "왼쪽/광대/x", "왼쪽/광대/y", "오른쪽/광대/x", "오른쪽/광대/y"])
  pd.set_option('display.max_columns', 100)
  pd.set_option('display.max_row', 100)
  print(point_dataframe.head())
  point_dataframe.to_csv(f"face_point_data.csv",encoding='utf-8-sig', mode='w')
  # point_dataframe = pd.DataFrame(np.array(point_data_list[0]))
  #
  # for i in point_data_list:
  #     point_dataframe.append(np.array(i))
  #
  # print(point_dataframe)
  # print(point_dataframe)
    # cv2.imshow('hi', annotated_image)  ###### 결과 사진 출력
    # cv2.waitKey(0)
    # cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
