import mediapipe as mp
import cv2
import numpy as np
import custom_drawing_utils
import time

mean_data = {
  '왼쪽 광대뼈 수직 비율'    : 0.566011,
  '오른쪽 광대뼈 수직 비율'  : 0.565264,
  '왼쪽 눈썹 수직 비율'     : 0.049600,
  '오른쪽 눈썹 수직 비율'    : 0.050270,
  '왼쪽눈 수직 비율'        : 0.065364,
  '오른쪽눈 수직 비율'      : 0.065270,
  '입 수직 비율'           : 0.105341,
  '입 수평 비율'           : 0.399008,
  '왼눈 수평 비율'         :  0.201380,
  '오른눈 수평 비율'       :  0.208207,
  '왼눈 오른쪽 각도'       : 28.862138,
  '왼눈 왼쪽 각도'         : 48.526707,
  '오른눈 오른쪽 각도'      : 45.777320,
  '오른눈 왼쪽 각도  '      : 28.398816,
  '입술 왼쪽 각도  '        : 28.896719,
  '입술 오른쪽 각도 '       : 4.834284
}
FACE_POINT = ( # 총 42개
  107, 55, 105, 52, 70, 46, 336, 285, 334, 282, 300, 276, 159, 144, 33, 133, 386, 373, 362, 263, 17, 61, 291, 0, 78,
  308, 14, 13,
  10, 297, 389, 356, 288, 378, 152, 149, 58, 127, 162, 67,
  50, 280)  # 10부터 시계방향 50,280(왼쪽 광대, 오른쪽 광대)
POSE_POINT = {
  'NOSE': 0,
  'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
  'LEFT_ELBOW' : 13, 'mp_holistic.PoseLandmark.RIGHT_ELBOW' : 14,
  'LEFT_WRIST' : 15, 'RIGHT_WRIST' : 16
}
FACE_RATE_DICT={"왼쪽 광대뼈 수직 비율": [39, 40, 39, 35], "오른쪽 광대뼈 수직 비율": [29, 41, 29, 33],
                 "왼쪽 눈썹 수직 비율": [3, 13, 39, 35], "오른쪽 눈썹 수직 비율": [9, 13, 29, 33],
                 "왼쪽눈 수직 비율": [12, 13, 39, 35], "오른쪽눈 수직 비율": [16, 17, 29, 33],
                 "왼눈 수평 비율": [14, 15, 31, 37], "오른눈 수평 비율": [18, 19, 31, 37],
                 "입 수직 비율": [20, 23, 28, 34], "입 수평 비율": [21, 22, 32, 36]}

FACE_ANGLE_DICT={"왼눈 오른쪽 각도": [12, 15, 13], "왼눈 왼쪽 각도": [12, 14, 13],
                 "오른눈 오른쪽 각도": [16, 19, 17],"오른눈 왼쪽 각도": [16, 18, 17],
                 "입술 왼쪽 각도": [27, 24, 26],"입술 오른쪽 각도": [27, 25, 26]}


def point_to_point_distance(p1:list, p2:list)->float:
  x = p2[0]- p1[0]
  y = p2[1]- p1[1]
  # z = p2[2]- p1[2]

  # return np.sqrt(np.power(x,2) + np.power(y,2) + np.power(z,2))
  if y < 0:
    return np.sqrt(np.power(x, 2) + np.power(y, 2)) * -1

  return np.sqrt(np.power(x, 2) + np.power(y, 2))

def angle_between(p1, p2)->float:  # 두점 사이의 각도:(getAngle3P 계산용) 시계 방향으로 계산한다. P1-(0,0)-P2의 각도를 시계방향으로
  ang1 = np.arctan2(*p1[::-1])
  ang2 = np.arctan2(*p2[::-1])
  res = np.rad2deg((ang1 - ang2) % (2 * np.pi))
  return res

def get_angle(p1:list, p2:list, p3:list)->float:
  pt1 = (p1[0] - p2[0], p1[1] - p2[1])
  pt2 = (p3[0] - p2[0], p3[1] - p2[1])
  angle = angle_between(pt1, pt2)
  angle = (angle + 360) % 360
  # if direction == "CCW":  # 시계방향
  if angle > 180:  # 시계방향
    angle = (360 - angle) % 360

  return angle

# def get_angle_face(name, img_point_dict_data):
#   angle_name_point_dict = {"왼눈 오른쪽 각도": [159, 133, 144], "왼눈 왼쪽 각도": [159, 33, 144], "오른눈 오른쪽 각도": [386, 263, 373],
#                            "오른눈 왼쪽 각도": [386, 362, 373], "입술 왼쪽 각도": [13, 78, 14],
#                            "입술 오른쪽 각도": [13, 308, 14]}
#
#   if name in angle_name_point_dict:
#     insert_point_list = angle_name_point_dict[name]
#     result_angle = angle_between(img_point_dict_data[insert_point_list[0]], img_point_dict_data[insert_point_list[1]],
#                                  img_point_dict_data[insert_point_list[2]])
#
#     return result_angle
#
# def angle_between_face(p1, p2, p3):  # 3점 사이 각도
#   x1, y1 = p1
#   x2, y2 = p2
#   x3, y3 = p3
#   deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360
#   deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360
#
#   result = deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)
#
#   if result > 180:
#     result = 360 - result
#
#   return result

def cvt_Landmark_to_list(landmark)-> list:
  x = landmark.x
  y = landmark.y
  # z = landmark.z

  # return [x, y, z]
  return [x, y]

def detect_head_down(pose_landmarks:list)->float:
  nose_to_sholuder_l = point_to_point_distance(pose_landmarks[0],pose_landmarks[1])
  nose_to_sholuder_c = point_to_point_distance(pose_landmarks[0],pose_landmarks[7])
  nose_to_sholuder_r = point_to_point_distance(pose_landmarks[0],pose_landmarks[2])

  l_elbow_angle = get_angle(pose_landmarks[1],pose_landmarks[3],pose_landmarks[5])
  r_elbow_angle = get_angle(pose_landmarks[2],pose_landmarks[4],pose_landmarks[6])

  print(f'코, 왼어깨:{nose_to_sholuder_l} | 코, 어깨가운데:{nose_to_sholuder_c} | 코, 오른어깨:{nose_to_sholuder_r} | 왼팔꿈치 각도:{l_elbow_angle} | 오른팔꿈치 각도:{r_elbow_angle}\n')

  score = 1.0

  if nose_to_sholuder_l < 0 or nose_to_sholuder_c < 0 or nose_to_sholuder_r < 0:  # 고개를 푹 숙인경우
    score -= 1

  ######## 어깨 가운데 #########
  if nose_to_sholuder_c < 0.15 and nose_to_sholuder_c > 0.13:
    print('살짝 숙임')
    score -= 0.5
  elif nose_to_sholuder_c <= 0.13:
    print('많이 숙임')
    score -= 10.0
  else:
    print('고개 안숙임')
    score += 1.0

  ######## 왼쪽 어깨 #########
  if nose_to_sholuder_l < 0.20 and nose_to_sholuder_l > 0.17:
    print('살짝 기울어짐')
    score -= 0.25
  elif nose_to_sholuder_l <= 0.17:
    print('많이 기울어짐')
    score -= 10
  else:
    print('안기울어짐')
    score += 0.5

  ######## 오른쪽 어깨 #########
  if nose_to_sholuder_r < 0.20 and nose_to_sholuder_r > 0.17:
    print('살짝 기울어짐')
    score -= 0.25
  elif nose_to_sholuder_r <= 0.17:
    print('많이 기울어짐')
    score -= 10
  else:
    print('안기울어짐')
    score += 0.5

  print(f'고개: {score}')
  return score

def detect_eye_closed(rate_angle_data:dict)->float:  # rate_angle_data ---> 얼굴 좌표로 얻은 비율과 각도
  score = 0
  keys_list = list(rate_angle_data.keys())
  # print(abs(values_gap["왼쪽눈 수직 비율"]))

  # 길이까지 같이 재야 정확하게 잴 수 있을듯
  eye_angle_sum = rate_angle_data[keys_list[-4]] + rate_angle_data[keys_list[-5]]
  if eye_angle_sum > 80:  # 집중
    score += 10
  elif eye_angle_sum > 70:  # 지루
    score -= 0.5
  else:  # 잠
    score -= 10
  # print(rate_angle_data[keys_list[-4]])
  # print(rate_angle_data[keys_list[-5]])
  print(f'눈점수: {score}')

  # result = ['']
  # if score >= 1.0:
  #   result[0] = 'A'
  # elif score >= 0:
  #   result[0] = 'B'
  # else:
  #   result[0] = 'C'
  #
  # return result[0]

  return score

def classify(face_landmarks:list, pose_landmarks:list)->float:
  score = 1.0

  head_down_score = detect_head_down(pose_landmarks)

  eye_closed_score = detect_eye_closed(process_data_rates(face_landmarks))

  score = head_down_score + eye_closed_score + score

  return score

def process_data_rates(face_landmarks:list) -> dict:

  current_img_processing_data = {}  # 현재 이미지 값 저장 변수

  for name in FACE_RATE_DICT.keys():
    if "비율" in name:
      current_img_processing_data[name] = rate_processing(FACE_RATE_DICT[name], face_landmarks)
    elif "각도" in name:
      # current_img_processing_data[name] = get_angle(name, face_landmarks) # 인자 안맞는거 오버로딩으로 해결을 할까--->안됨
      current_img_processing_data[name] = get_angle(face_landmarks[FACE_ANGLE_DICT[name][0]],face_landmarks[FACE_ANGLE_DICT[name][1]],face_landmarks[FACE_ANGLE_DICT[name][2]])  # 인자 안맞는거 오버로딩으로 해결을 할까--->안됨

    else:
      print("오류?")

  # print(current_img_processing_data)
  return current_img_processing_data

def rate_processing(idx:list, face_landmarks:list) -> float:
  # rate_name_point_dict = {"왼쪽 광대뼈 수직 비율": [67, 50, 67, 149], "오른쪽 광대뼈 수직 비율": [297, 280, 297, 378],
  #                         "왼쪽 눈썹 수직 비율": [52, 144, 67, 149], "오른쪽 눈썹 수직 비율": [282, 373, 297, 378],
  #                         "왼쪽눈 수직 비율": [159, 144, 67, 149], "오른쪽눈 수직 비율": [386, 373, 297, 378],
  #                         "입 수직 비율": [17, 0, 10, 152], "입 수평 비율": [61, 291, 288, 58], "왼눈 수평 비율": [33, 133, 356, 127],
  #                         "오른눈 수평 비율": [362, 263, 356, 127]}

  first_rate = point_to_point_distance(face_landmarks[idx[0]],face_landmarks[idx[1]])
  second_rate = point_to_point_distance(face_landmarks[idx[2]],face_landmarks[idx[3]])
  result_rate = first_rate / second_rate

  return result_rate

# -----------------------------------------------
mp_drawing = custom_drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(model_complexity = 2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      # continue
      break

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = holistic.process(image)

    pose_data = []
    face_data = []

    try :

      for idx in POSE_POINT.values():
        pose_data.append(cvt_Landmark_to_list(results.pose_landmarks.landmark[idx])) # list[list] 형태를 가짐

      center_of_shoulder = [(pose_data[1][0] + pose_data[2][0])/2, (pose_data[1][1] + pose_data[2][1])/2]
      pose_data.append(center_of_shoulder)

      for idx in FACE_POINT:
        face_data.append(cvt_Landmark_to_list(results.face_landmarks.landmark[idx]))# list[list] 형태를 가짐

    except Exception as e:
      print('Cannot find feature')
      print(e)
      continue

    # img_processing_data = process_data_rates(results.face_landmarks.landmark) # landmark의 값은 .x .y 같은 방식으로 접근하면된다. 이제 dict를 list를 바꾸거나 간단화 작업을 해야함

    # new_data = np.array([nose[0], nose[1],
    #                      l_shoulder[0], r_shoulder[0], l_elbow[0], r_elbow[0], l_wrist[0], r_wrist[0],
    #                      l_shoulder[1], r_shoulder[1], l_elbow[1], r_elbow[1], l_wrist[1], r_wrist[1],
    #                      point_to_point_distance(nose, l_shoulder), point_to_point_distance(nose, center_of_shoulder), point_to_point_distance(nose, r_shoulder),
    #                      angle_processing(l_shoulder, l_elbow, l_wrist), angle_processing(r_shoulder, r_elbow, r_wrist)
    #                      ], dtype=np.float32)

    # new_data = new_data.reshape(1, -1)

    predict_class = classify(face_data, pose_data)


    print(predict_class)
    if predict_class < 0:
      print("Sleep!\n")
    else:
      print("Concentrate!\n")

    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # mp_drawing.draw_landmarks(
    #     image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
    # mp_drawing.draw_landmarks(
    #     image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # mp_drawing.draw_landmarks(
    #     image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)


    image = cv2.resize(image, (720, 480))
    cv2.imshow('MediaPipe Holistic', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

    if cv2.waitKey(1) & 0xFF == 32:
      time.sleep(10)
cap.release()
