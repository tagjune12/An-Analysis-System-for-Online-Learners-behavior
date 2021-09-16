import numpy as np

FACE_POINT = (
  107, 55, 105, 52, 70, 46, 336, 285, 334, 282, 300, 276, 159, 144, 33, 133, 386, 373, 362, 263, 17, 61, 291, 0, 78,
  308, 14, 13,
  10, 297, 389, 356, 288, 378, 152, 149, 58, 127, 162, 67,
  50, 280)  # 10부터 시계방향 50,280(왼쪽 광대, 오른쪽 광대)
POSE_POINT = {
  'NOSE': 0,
  'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
  'LEFT_ELBOW' : 13, 'RIGHT_ELBOW' : 14,
  'LEFT_WRIST' : 15, 'RIGHT_WRIST' : 16
}
FACE_RATE_DICT={"왼쪽 광대뼈 수직 비율": [39, 40, 39, 35], "오른쪽 광대뼈 수직 비율": [29, 41, 29, 33],
                 "왼쪽 눈썹 수직 비율": [3, 13, 39, 35], "오른쪽 눈썹 수직 비율": [9, 13, 29, 33],
                 "왼쪽눈 수직 비율": [12, 13, 39, 35], "오른쪽눈 수직 비율": [16, 17, 29, 33],
                 "왼눈 수평 비율": [14, 15, 31, 37], "오른눈 수평 비율": [18, 19, 31, 37],
                 "입 수직 비율": [20, 23, 28, 34], "입 수평 비율": [21, 22, 32, 36],
                 "왼눈 오른쪽 각도": [12, 15, 13], "왼눈 왼쪽 각도": [12, 14, 13],
                 "오른눈 오른쪽 각도": [16, 19, 17],"오른눈 왼쪽 각도": [16, 18, 17],
                 "입술 왼쪽 각도": [27, 24, 26],"입술 오른쪽 각도": [27, 25, 26]
                }

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

def cvt_Landmark_to_list(landmark)-> list:
  x = landmark.x
  y = landmark.y
  # z = landmark.z

  # return [x, y, z]
  return [x, y]

def preprocess(features):

  pose_data = []
  face_data = []

  # try:
  #
  #   for idx in POSE_POINT.values():
  #     pose_data.append(cvt_Landmark_to_list(features.pose_landmarks.landmark[idx])) # list[list] 형태를 가짐
  #
  #   center_of_shoulder = [(pose_data[1][0] + pose_data[2][0])/2, (pose_data[1][1] + pose_data[2][1])/2]
  #   pose_data.append(center_of_shoulder)
  #
  # except Exception as e:
  #   print('Cannot find some pose features')
  #   print(e)
  #   pose_data.clear()
  #
  # try:
  #   for idx in FACE_POINT:
  #     face_data.append(cvt_Landmark_to_list(features.face_landmarks.landmark[idx]))  # list[list] 형태를 가짐
  #
  # except Exception as e:
  #   print('Cannot find some face features')
  #   print(e)
  #   face_data.clear()

  for feature in features:
    try:

      for idx in POSE_POINT.values():
        pose_data.append(cvt_Landmark_to_list(feature.pose_landmarks.landmark[idx]))  # list[list] 형태를 가짐

      center_of_shoulder = [(pose_data[1][0] + pose_data[2][0]) / 2, (pose_data[1][1] + pose_data[2][1]) / 2]
      pose_data.append(center_of_shoulder)

    except Exception as e:
      print('Cannot find some pose features')
      print(e)
      pose_data.clear()

    try:
      for idx in FACE_POINT:
        face_data.append(cvt_Landmark_to_list(feature.face_landmarks.landmark[idx]))  # list[list] 형태를 가짐

    except Exception as e:
      print('Cannot find some face features')
      print(e)
      face_data.clear()

  return  face_data, pose_data

def classify(features:list)->float:

  face_landmarks, pose_landmarks = preprocess(features)
  if not face_landmarks or not pose_landmarks :
    return None

  score = 1.0

  head_down_score = detect_head_down(pose_landmarks)

  eye_closed_score = detect_eye_closed(process_data_rates(face_landmarks))

  score = head_down_score + eye_closed_score + score

  # print('Log from classify of classification_module')
  return score

def detect_head_down(pose_landmarks:list)->float:
  nose_to_sholuder_l = point_to_point_distance(pose_landmarks[0],pose_landmarks[1])
  nose_to_sholuder_c = point_to_point_distance(pose_landmarks[0],pose_landmarks[7])
  nose_to_sholuder_r = point_to_point_distance(pose_landmarks[0],pose_landmarks[2])

  l_elbow_angle = get_angle(pose_landmarks[1],pose_landmarks[3],pose_landmarks[5])
  r_elbow_angle = get_angle(pose_landmarks[2],pose_landmarks[4],pose_landmarks[6])

  # print(f'코, 왼어깨:{nose_to_sholuder_l} | 코, 어깨가운데:{nose_to_sholuder_c} | 코, 오른어깨:{nose_to_sholuder_r} | 왼팔꿈치 각도:{l_elbow_angle} | 오른팔꿈치 각도:{r_elbow_angle}\n')

  score = 1.0

  if nose_to_sholuder_l < 0 or nose_to_sholuder_c < 0 or nose_to_sholuder_r < 0:  # 고개를 푹 숙인경우
    score -= 1

  ######## 어깨 가운데 #########
  if nose_to_sholuder_c < 0.15 and nose_to_sholuder_c > 0.13:
    # print('살짝 숙임')
    score -= 0.5
  elif nose_to_sholuder_c <= 0.13:
    # print('많이 숙임')
    score -= 10.0
  else:
    # print('고개 안숙임')
    score += 1.0

  ######## 왼쪽 어깨 #########
  if nose_to_sholuder_l < 0.20 and nose_to_sholuder_l > 0.17:
    # print('살짝 기울어짐')
    score -= 0.25
  elif nose_to_sholuder_l <= 0.17:
    # print('많이 기울어짐')
    score -= 10
  else:
    # print('안기울어짐')
    score += 0.5

  ######## 오른쪽 어깨 #########
  if nose_to_sholuder_r < 0.20 and nose_to_sholuder_r > 0.17:
    # print('살짝 기울어짐')
    score -= 0.25
  elif nose_to_sholuder_r <= 0.17:
    # print('많이 기울어짐')
    score -= 10
  else:
    # print('안기울어짐')
    score += 0.5

  # print(f'고개: {score}')

  # print('Log from detect_head_down of classification_module')
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

  # print(f'눈점수: {score}')
  # print('Log from detect_eye_closed of classification_module')

  return score

def process_data_rates(face_landmarks:list) -> dict:

  current_img_processing_data = {}  # 현재 이미지 값 저장 변수

  for name in FACE_RATE_DICT.keys():
    if "비율" in name:
      current_img_processing_data[name] = rate_processing(FACE_RATE_DICT[name], face_landmarks)
    elif "각도" in name:
      current_img_processing_data[name] = get_angle(face_landmarks[FACE_RATE_DICT[name][0]],face_landmarks[FACE_RATE_DICT[name][1]],face_landmarks[FACE_RATE_DICT[name][2]])  # 인자 안맞는거 오버로딩으로 해결을 할까--->안됨
      # print(f'각도{name}:{current_img_processing_data[name]}')

    else:
      print("오류?")

  return current_img_processing_data

def rate_processing(idx:list, face_landmarks:list) -> float:

  first_rate = point_to_point_distance(face_landmarks[idx[0]],face_landmarks[idx[1]])
  second_rate = point_to_point_distance(face_landmarks[idx[2]],face_landmarks[idx[3]])
  result_rate = first_rate / second_rate

  return result_rate







