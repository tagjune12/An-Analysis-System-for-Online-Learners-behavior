import mediapipe as mp
import cv2
import numpy as np
import custom_drawing_utils

def length(p1:list, p2:list):
  x = p2[0]- p1[0]
  y = p2[1]- p1[1]

  if y < 0:
    return np.sqrt(np.power(x, 2) + np.power(y, 2)) * -1

  return np.sqrt(np.power(x, 2) + np.power(y, 2))

def angle_between(p1, p2):  # 두점 사이의 각도:(getAngle3P 계산용) 시계 방향으로 계산한다. P1-(0,0)-P2의 각도를 시계방향으로
  ang1 = np.arctan2(*p1[::-1])
  ang2 = np.arctan2(*p2[::-1])
  res = np.rad2deg((ang1 - ang2) % (2 * np.pi))
  return res

def angle(p1:list,p2:list,p3:list):

  pt1 = (p1[0] - p2[0], p1[1] - p2[1])
  pt2 = (p3[0] - p2[0], p3[1] - p2[1])
  angle = angle_between(pt1, pt2)
  angle = (angle + 360) % 360

  if angle > 180:  # 시계방향
    angle = (360 - angle) % 360

  return angle

def cvt_Landmark_to_list(landmark):
  x = landmark.x
  y = landmark.y

  return [x, y]


def calc_score(new_data):
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



mp_drawing = custom_drawing_utils
mp_holistic = mp.solutions.holistic

# For webcam input:
# cap = cv2.VideoCapture('./Videos/상민(자는중).mp4') # 동영상 경로
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

    try :

      nose = cvt_Landmark_to_list(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE])
      l_shoulder = cvt_Landmark_to_list(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER])
      r_shoulder = cvt_Landmark_to_list(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER])
      l_elbow = cvt_Landmark_to_list(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW])
      r_elbow = cvt_Landmark_to_list(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW])
      l_wrist = cvt_Landmark_to_list(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST])
      r_wrist = cvt_Landmark_to_list(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST])
      center_of_shoulder = [(l_shoulder[0] + r_shoulder[0])/2, (l_shoulder[1] + r_shoulder[1])/2]
    except:
      print('Cannot find feature')
      continue

    new_data = np.array([nose[0], nose[1],
                         l_shoulder[0], r_shoulder[0], l_elbow[0], r_elbow[0], l_wrist[0], r_wrist[0],
                         l_shoulder[1], r_shoulder[1], l_elbow[1], r_elbow[1], l_wrist[1], r_wrist[1],
                         length(nose, l_shoulder), length(nose, center_of_shoulder), length(nose, r_shoulder),
                         angle(l_shoulder, l_elbow, l_wrist), angle(r_shoulder, r_elbow, r_wrist)
                         ], dtype=np.float32)

    new_data = new_data.reshape(1, -1)

    predict_class = calc_score(new_data)

    l_elbow_visible = results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].visibility
    r_elbow_visible = results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].visibility
    l_wrist_visible = results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].visibility
    r_wrist_visible = results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].visibility


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
cap.release()
