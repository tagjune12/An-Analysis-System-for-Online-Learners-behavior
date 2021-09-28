import numpy as np
# 미디어 파이프에서 명시하는 우리가 실제로 필요한 특징점들 인덱스 by.인석
FACE_POINT = (
  107, 55, 105, 52, 70, 46, 336, 285, 334, 282, 300, 276, 159, 144, 33, 133, 386, 373, 362, 263, 17, 61, 291, 0, 78,
  308, 14, 13,
  10, 297, 389, 356, 288, 378, 152, 149, 58, 127, 162, 67,
  50, 280)  # 10부터 시계방향 50,280(왼쪽 광대, 오른쪽 광대)

# Mediapipe에서 지정한 특징점 인덱스중 필요한것만 구조화 by.택준
POSE_POINT = {
  'NOSE': 0,
  'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
  'LEFT_ELBOW' : 13, 'RIGHT_ELBOW' : 14,
  'LEFT_WRIST' : 15, 'RIGHT_WRIST' : 16
}

# 얼굴 특징점 일부만 사용하기 위한 딕셔너리 값 by.택준
FACE_RATE_DICT={"왼쪽 광대뼈 수직 비율": [39, 40, 39, 35], "오른쪽 광대뼈 수직 비율": [29, 41, 29, 33],
                 "왼쪽 눈썹 수직 비율": [3, 13, 39, 35], "오른쪽 눈썹 수직 비율": [9, 13, 29, 33],
                 "왼쪽눈 수직 비율": [12, 13, 39, 35], "오른쪽눈 수직 비율": [16, 17, 29, 33],
                 "왼눈 수평 비율": [14, 15, 31, 37], "오른눈 수평 비율": [18, 19, 31, 37],
                 "입 수직 비율": [20, 23, 28, 34], "입 수평 비율": [21, 22, 32, 36],
                 "왼눈 오른쪽 각도": [12, 15, 13], "왼눈 왼쪽 각도": [12, 14, 13],
                 "오른눈 오른쪽 각도": [16, 19, 17],"오른눈 왼쪽 각도": [16, 18, 17],
                 "입술 왼쪽 각도": [27, 24, 26],"입술 오른쪽 각도": [27, 25, 26]
                }
# 두 점사이의 거리를 구하는 함수 by.택준
def point_to_point_distance(p1:list, p2:list)->float: # p1, p2는 list[list] 형태여야한다.
  x = p2[0]- p1[0]
  y = p2[1]- p1[1]

  if y < 0:
    return np.sqrt(np.power(x, 2) + np.power(y, 2)) * -1

  return np.sqrt(np.power(x, 2) + np.power(y, 2))

# 두 벡터를 이용하여 각도를 계산하는 함수 by.택준
def angle_between(p1, p2)->float:  # 두점 사이의 각도:(getAngle3P 계산용) 시계 방향으로 계산한다. P1-(0,0)-P2의 각도를 시계방향으로
  ang1 = np.arctan2(*p1[::-1])
  ang2 = np.arctan2(*p2[::-1])
  res = np.rad2deg((ang1 - ang2) % (2 * np.pi))
  return res

# 세 점을 이용하여 각도를 계산하는 함수 by.택준
def get_angle(p1:list, p2:list, p3:list)->float:
  pt1 = (p1[0] - p2[0], p1[1] - p2[1])
  pt2 = (p3[0] - p2[0], p3[1] - p2[1])
  angle = angle_between(pt1, pt2)
  angle = (angle + 360) % 360

  if angle > 180:  # 시계방향
    angle = (360 - angle) % 360

  return angle

# 미디어 파이프의 output을 list로 변환하는 함수 by.택준
def cvt_Landmark_to_list(landmark)-> list:
  x = landmark.x
  y = landmark.y

  return [x, y]

def preprocess(features):

  pose_datas = []
  face_datas = []
  visibilities = []

  for feature in features: # feature => list

    pose_data = []
    face_data = []
    visibility = []

    # 얼굴
    try:
      for idx in POSE_POINT.values():
        pose_data.append(cvt_Landmark_to_list(feature.pose_landmarks.landmark[idx]))  # pose_data => list[list] 형태를 가짐

      center_of_shoulder = [(pose_data[1][0] + pose_data[2][0]) / 2, (pose_data[1][1] + pose_data[2][1]) / 2]
      pose_data.append(center_of_shoulder)

    except Exception as e:
      print('Cannot find some pose features')
      print(e)
      pose_data.append('None')
    # 신체
    try:
      for idx in FACE_POINT:
        face_data.append(cvt_Landmark_to_list(feature.face_landmarks.landmark[idx]))  # face_data => list[list] 형태를 가짐

    except Exception as e:
      print('Cannot find some face features')
      print(e)
      face_data.append('None')

    try:
      visibility = [feature.pose_landmarks.landmark[15].visibility,feature.pose_landmarks.landmark[16].visibility]

    except Exception as e:
      print('Error from visiblity')
      print(e)
      visibility.append('None')


    face_datas.append(face_data)
    pose_datas.append(pose_data)
    visibilities.append(visibility)

  print(f'Log from preprocess{np.shape(face_datas)}, {np.shape(pose_datas)}')
  return  face_datas, pose_datas, visibilities # list[list], list[list] # (2,8,2) (2,42,2) // (사람수,8,2) (사람수,42,2)

# 얼굴과 신체의 특징점을 이용하여 태도를 분석하고 점수화 시키는 함수 by.택준
def classify(features:list)->float:

  face_landmarks, pose_landmarks, visibilities = preprocess(features)

  state = []

  for face_landmark, pose_landmark, visibility in zip(face_landmarks, pose_landmarks, visibilities):
    # if not face_landmark or not pose_landmark :
    #   return None
    if type(face_landmark[0]) == type('None'):
      state.append('판별 불가능')
      continue

    if type(pose_landmark[0]) == type('None'):
      state.append('판별 불가능')
      continue

    if type(visibility[0]) == type('None'):
      state.append('판별 불가능')
      continue

    head_down_score = detect_head_down(pose_landmark, face_landmark, visibility)

    # eye_closed_score = detect_eye_closed(process_data_rates(face_landmark))
    eye_closed_score = detect_eye_closed(process_data_rates(face_landmark), pose_landmark)  # string

    # scores.append(head_down_score + eye_closed_score + score)

    # print('Log from classify of classification_module')

    if eye_closed_score['눈'] == '눈 감고 있음' or head_down_score['머리'] == '푹 숙임':

        state.append("잠")

    elif eye_closed_score['눈'] == '눈 흐리게 떠있음':
        if head_down_score['손목'] == '턱 굄' and (
                any(keyword in head_down_score['왼쪽 어깨'] for keyword in ["살짝", "많이"]) or any(
            keyword in head_down_score['오른쪽 어깨'] for keyword in ["살짝", "많이"])):
            state.append('지루함')
        else:
          state.append('집중')

    else:
      state.append('집중')

  return state

# 얼굴 특징점들을 이용하여 고개가 얼마나 숙여졌는지 판단하여 점수화 하는 함수 by.택준
def detect_head_down(pose_landmarks:list, face_landmarks:list, visibilities:list)->float:
  pose_result_dict = {}  # 포즈 판별 값 저장 할 딕셔너리

  print(f'detect_head_down: {np.shape(pose_landmarks)} {np.shape(face_landmarks)} {np.shape}')

  nose_to_sholuder_l = point_to_point_distance(pose_landmarks[0],pose_landmarks[1])
  nose_to_sholuder_c = point_to_point_distance(pose_landmarks[0],pose_landmarks[7])
  nose_to_sholuder_r = point_to_point_distance(pose_landmarks[0],pose_landmarks[2])


  # print(f'코, 왼어깨:{nose_to_sholuder_l} | 코, 어깨가운데:{nose_to_sholuder_c} | 코, 오른어깨:{nose_to_sholuder_r} | 왼팔꿈치 각도:{l_elbow_angle} | 오른팔꿈치 각도:{r_elbow_angle}\n')

  score = 1.0

  if nose_to_sholuder_l < 0 or nose_to_sholuder_c < 0 or nose_to_sholuder_r < 0:  # 고개를 푹 숙인경우
    pose_result_dict['머리'] = "푹 숙임"
    score -= 1

  ######## 어깨 가운데 #########
  if nose_to_sholuder_c < 0.15 and nose_to_sholuder_c > 0.13:
    print('살짝 숙임')
    pose_result_dict['머리'] = "살짝 숙임"
    score -= 0.5
  elif nose_to_sholuder_c <= 0.13:
    print('많이 숙임')
    pose_result_dict['머리'] = "많이 숙임"
    score -= 10.0
  else:
    print('고개 안숙임')
    pose_result_dict['머리'] = "안 숙임"
    score += 1.0

  ######## 왼쪽 어깨 #########
  if nose_to_sholuder_l < 0.20 and nose_to_sholuder_l > 0.17:
    print('왼쪽 살짝 기울어짐')
    pose_result_dict['왼쪽 어깨'] = "살짝 기울어짐"
    score -= 0.25
  elif nose_to_sholuder_l <= 0.17:
    print('왼쪽 많이 기울어짐')
    pose_result_dict['왼쪽 어깨'] = "많이 기울어짐"
    score -= 10
  else:
    print('왼쪽 안기울어짐')
    pose_result_dict['왼쪽 어깨'] = "안 기울어짐"
    score += 0.5

  ######## 오른쪽 어깨 #########
  if nose_to_sholuder_r < 0.20 and nose_to_sholuder_r > 0.17:
    print('오른쪽 살짝 기울어짐')
    pose_result_dict['오른쪽 어깨'] = "살짝 기울어짐"
    score -= 0.25
  elif nose_to_sholuder_r <= 0.17:
    print('오른쪽 많이 기울어짐')
    pose_result_dict['오른쪽 어깨'] = "많이 기울어짐"
    score -= 10
  else:
    print('오른쪽 안기울어짐')
    pose_result_dict['오른쪽 어깨'] = "안 기울어짐"
    score += 0.5

  ############손목 검출 여부 ###########
  l_wrist_visible = visibilities[0]
  r_wrist_visible = visibilities[1]
  # print(f"현재 l_wrist_visible = {l_wrist_visible}")
  # print(f"현재 r_wrist_visible = {r_wrist_visible}")

  pose_result_dict['손목'] = "안 굄"

  if l_wrist_visible > 0.50 and (l_wrist_visible > r_wrist_visible):  # 0.70
    # print("왼손 검출")
    # print("손부터 뺨까지 거리", point_to_point_distance(pose_landmarks[5], face_landmarks[36]))
    if abs(point_to_point_distance(pose_landmarks[5], face_landmarks[36])) < 0.28:
      # print("턱 괴고 있음@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
      pose_result_dict['손목'] = "턱 굄"
  elif r_wrist_visible > 0.50 and (r_wrist_visible > l_wrist_visible):  # 0.70
    # print("오른손 검출")
    # print("손부터 뺨까지 거리", point_to_point_distance(pose_landmarks[6], face_landmarks[32]))
    if abs(point_to_point_distance(pose_landmarks[6], face_landmarks[32])) < 0.28:
      # print("턱 괴고 있음@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
      pose_result_dict['손목'] = "턱 굄"



  # print(f'고개: {score}')
  return pose_result_dict

# 눈이 감겼는지 판단하는 함수
def detect_eye_closed(rate_angle_data:dict, pose_landmarks)->float:  # rate_angle_data ---> 얼굴 좌표로 얻은 비율과 각도
  face_result_dict = {}  # 얼굴 검출 값 저장 할 딕셔너리

  nose_to_sholuder_c = point_to_point_distance(pose_landmarks[0], pose_landmarks[7])

  if nose_to_sholuder_c >= 0.190:
    top_base_score, middle_base_score = 31, 25

  elif nose_to_sholuder_c >= 0.182:
    top_base_score, middle_base_score = 30, 25

  elif nose_to_sholuder_c >= 0.155:  # 이경우엔 눈흐림 기준 -4 ?? 뜰때34 33, 33 32, 35 33 ㅡ  흐리게  25 24, 27 26, 27 25 ㅡ 감을때 23 22, 22 21, 23 22
    top_base_score, middle_base_score = 30, 25

  else:  # 이경우엔 눈흐림 기준 -3 ??뜰때 31 30, 30 29, 30, 29 ㅡ감을때 24 24, 22 21, 22 21, 24 23
    top_base_score, middle_base_score = 29, 25.5

  if top_base_score < rate_angle_data['왼눈 오른쪽 각도'] or top_base_score < rate_angle_data['오른눈 왼쪽 각도']:
    print('눈 떠있음')
    face_result_dict['눈'] = '눈 떠있음'

  elif middle_base_score < rate_angle_data['왼눈 오른쪽 각도'] or middle_base_score < rate_angle_data['오른눈 왼쪽 각도']:
    print('눈 흐리게 떠있음')
    face_result_dict['눈'] = '눈 흐리게 떠있음'

  else:
    print('눈 감고 있음')
    face_result_dict['눈'] = '눈 감고 있음'

  ##################하품 #######
  if rate_angle_data['입술 왼쪽 각도'] > 45 or rate_angle_data['입술 왼쪽 각도'] > 45:
    print('하품')
    face_result_dict['입'] = '하품'
  else:
    face_result_dict['입'] = '정상'


  # print(f"\n\n현재 고개숙임 정도 ##########{point_to_point_distance(pose_landmarks[0], pose_landmarks[7])}######\n\n")


  return face_result_dict

# 얼굴 일부 특징점을 이용하여 비율을 구하고 눈의 각도를 이용하여 눈이 감겼는지 판단하기 위한 데이터를 생성
def process_data_rates(face_landmarks:list) -> dict:

  current_img_processing_data = {}  # 현재 이미지 값 저장 변수

  for name in FACE_RATE_DICT.keys():
    # 얼굴의 일부 특징점을 이용하여 특징점간 거리의 비율을 구하고 데이터를 current_img_processing_data에 저장
    if "비율" in name:
      current_img_processing_data[name] = rate_processing(FACE_RATE_DICT[name], face_landmarks)
    # 눈이 얼마나 감겼는지 각도를 계한사여 데이터를 current_img_processing_data에 저장
    elif "각도" in name:
      current_img_processing_data[name] = get_angle(face_landmarks[FACE_RATE_DICT[name][0]],face_landmarks[FACE_RATE_DICT[name][1]],face_landmarks[FACE_RATE_DICT[name][2]])


    else:
      print("오류?")

  return current_img_processing_data

# 얼굴의 일부 특징점들의 값을 가져와 비율을 계산하는 함수
def rate_processing(idx:list, face_landmarks:list) -> float:

  first_rate = point_to_point_distance(face_landmarks[idx[0]],face_landmarks[idx[1]])
  second_rate = point_to_point_distance(face_landmarks[idx[2]],face_landmarks[idx[3]])
  result_rate = first_rate / second_rate

  return result_rate







