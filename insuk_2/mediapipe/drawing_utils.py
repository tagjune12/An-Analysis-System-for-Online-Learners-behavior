# Copyright 2020 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MediaPipe solution drawing utils."""

import math
from typing import List, Tuple, Union

import cv2
import dataclasses
import numpy as np
import time         ####

from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import location_data_pb2
from mediapipe.framework.formats import landmark_pb2

PRESENCE_THRESHOLD = 0.5
RGB_CHANNELS = 3
RED_COLOR = (0, 0, 255)
VISIBILITY_THRESHOLD = 0.5



INSUK_POINT_LEFT_EYEBROW = (107,55,105,52,70,46)#미간을 기준으로 밖으로 안쪽 위, 아래 가운데 위, 아래 바깥 위, 아래 순
INSUK_POINT_RIGHT_EYEPROW = (336,285,334,282,300,276)#미간을 기준으로 밖으로 안쪽 위, 아래 가운데 위, 아래 바깥 위, 아래 순
INSUK_POINT_LEFT_EYE = (159,144,33,133)#12시, 6시, 9시, 3시
INSUK_POINT_RIGHT_EYE = (386,373,362,263)# 12시, 6시, 9시, 3시
INSUK_POINT_LIP = (17,61,291,0,78,308,14,13)#바깥 맨아래 가운데17 왼 61 오291 맨 위 가운데 0 #안쪽 왼 78 오 308 아래가운데 14 위가운데13
POINT = (107,55,105,52,70,46,336,285,334,282,300,276,159,144,33,133,386,373,362,263,17,61,291,0,78,308,14,13,
          10, 297, 389, 356, 288, 378, 152, 149, 58, 127, 162, 67,
         50, 280)# 10(12방향)부터 시계방향 50,280(왼쪽 광대, 오른쪽 광대)
#광대 50, 280



@dataclasses.dataclass
class DrawingSpec:
  # Color for drawing the annotation. Default to the green color.
  color: Tuple[int, int, int] = (0, 255, 0)
  # Thickness for drawing the annotation. Default to 2 pixels.
  thickness: int = 2
  # Circle radius. Default to 2 pixels.
  circle_radius: int = 2

def point_to_point_distance(idx_to_coordinates, image, POINT=POINT):########################################### 두점 거리 ############
  left_eye_12to6 = math.sqrt(math.pow((idx_to_coordinates[159][0] - idx_to_coordinates[144][0]), 2) + math.pow((idx_to_coordinates[159][1] - idx_to_coordinates[144][1]), 2))
  left_eye_9to3 = math.sqrt(math.pow((idx_to_coordinates[33][0] - idx_to_coordinates[133][0]), 2) + math.pow((idx_to_coordinates[33][1] - idx_to_coordinates[133][1]), 2))
  right_eye_12to6 = math.sqrt(math.pow((idx_to_coordinates[386][0] - idx_to_coordinates[373][0]), 2) + math.pow((idx_to_coordinates[386][1] - idx_to_coordinates[373][1]), 2))
  right_eye_9to3 = math.sqrt(math.pow((idx_to_coordinates[362][0] - idx_to_coordinates[263][0]), 2) + math.pow((idx_to_coordinates[362][1] - idx_to_coordinates[263][1]), 2))
  lip_12to6 = math.sqrt(math.pow((idx_to_coordinates[13][0] - idx_to_coordinates[14][0]), 2) + math.pow((idx_to_coordinates[13][1] - idx_to_coordinates[14][1]), 2))
  lip_9to3 = math.sqrt(math.pow((idx_to_coordinates[78][0] - idx_to_coordinates[308][0]), 2) + math.pow((idx_to_coordinates[78][1] - idx_to_coordinates[308][1]), 2))

  pointlist = [left_eye_12to6,left_eye_9to3,right_eye_12to6,right_eye_9to3,lip_12to6,lip_9to3]
  namelist = ["left_eye_12to6", "left_eye_9to3", "right_eye_12to6", "right_eye_9to3", "lip_12to6", "lip_9to3"]

  for index, (i, name) in enumerate(zip(pointlist,namelist),1):
    cv2.putText(image,f"{name}: ({i})",
                org=(0, 25+(index*20)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0))


def _normalized_to_pixel_coordinates(#픽셀 좌표로 정규화
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:#화살표 -> 반환되는 형식을 나타냄
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:# 벨류 값이 0보다 크고나 0이랑 비슷하고, 1보다 작거나 1보다 비슷할때  true
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and #is_valid_normalized_value함수를 만족하지않을대 flase
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1) #is_valid_normalized_value함수를 만족할 때 *min: 주어진 자료형에서 최소값 반환
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)#*math.floor 가장 가까운 정수로 내림
  return x_px, y_px


# def draw_detection(
#     image: np.ndarray,
#     detection: detection_pb2.Detection,
#     keypoint_drawing_spec: DrawingSpec = DrawingSpec(color=RED_COLOR),
#     bbox_drawing_spec: DrawingSpec = DrawingSpec()):
#   """Draws the detction bounding box and keypoints on the image.
#
#   Args:
#     image: A three channel RGB image represented as numpy ndarray.
#     detection: A detection proto message to be annotated on the image.
#     keypoint_drawing_spec: A DrawingSpec object that specifies the keypoints'
#       drawing settings such as color, line thickness, and circle radius.
#     bbox_drawing_spec: A DrawingSpec object that specifies the bounding box's
#       drawing settings such as color and line thickness.
#
#   Raises:
#     ValueError: If one of the followings:
#       a) If the input image is not three channel RGB.
#       b) If the location data is not relative data.
#   """
#   if not detection.location_data:
#     return
#   if image.shape[2] != RGB_CHANNELS:
#     raise ValueError('Input image must contain three channel rgb data.')
#   image_rows, image_cols, _ = image.shape
#
#   location = detection.location_data
#   if location.format != location_data_pb2.LocationData.RELATIVE_BOUNDING_BOX:
#     raise ValueError(
#         'LocationData must be relative for this drawing funtion to work.')
#   # Draws keypoints.
#   for keypoint in location.relative_keypoints:
#     keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
#                                                    image_cols, image_rows)
#     cv2.circle(image, keypoint_px, keypoint_drawing_spec.circle_radius,
#                keypoint_drawing_spec.color, keypoint_drawing_spec.thickness)
#   # Draws bounding box if exists.
#   if not location.HasField('relative_bounding_box'):
#     return
#   relative_bounding_box = location.relative_bounding_box
#   rect_start_point = _normalized_to_pixel_coordinates(
#       relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
#       image_rows)
#   rect_end_point = _normalized_to_pixel_coordinates(
#       relative_bounding_box.xmin + relative_bounding_box.width,
#       relative_bounding_box.ymin + +relative_bounding_box.height, image_cols,
#       image_rows)
#   cv2.rectangle(image, rect_start_point, rect_end_point,
#                 bbox_drawing_spec.color, bbox_drawing_spec.thickness)


def draw_landmarks(
    image: np.ndarray,
    landmark_list: landmark_pb2.NormalizedLandmarkList,
    connections: List[Tuple[int, int]] = None,
    landmark_drawing_spec: DrawingSpec = DrawingSpec(color=RED_COLOR),
    connection_drawing_spec: DrawingSpec = DrawingSpec()):
  """Draws the landmarks and the connections on the image.

  Args:
    image: A three channel RGB image represented as numpy ndarray.
    landmark_list: A normalized landmark list proto message to be annotated on
      the image.
    connections: A list of landmark index tuples that specifies how landmarks to
      be connected in the drawing.
    landmark_drawing_spec: A DrawingSpec object that specifies the landmarks'
      drawing settings such as color, line thickness, and circle radius.
    connection_drawing_spec: A DrawingSpec object that specifies the
      connections' drawing settings such as color and line thickness.

  Raises:
    ValueError: If one of the followings:
      a) If the input image is not three channel RGB.
      b) If any connetions contain invalid landmark index.
  """
  if not landmark_list:
    return
  if image.shape[2] != RGB_CHANNELS:# 3채널 아니면 에러
    raise ValueError('Input image must contain three channel rgb data.')
  image_rows, image_cols, _ = image.shape # 가로 세로 분리
  idx_to_coordinates = {}
  for idx, landmark in enumerate(landmark_list.landmark):
    if ((landmark.HasField('visibility') and
         landmark.visibility < VISIBILITY_THRESHOLD) or
        (landmark.HasField('presence') and
         landmark.presence < PRESENCE_THRESHOLD)):
      continue
    landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y, # 정규화 x y 반환
                                                   image_cols, image_rows)
    if landmark_px:# 값이 존재하면 idx_to_coordinates에 값 넣어줌
      idx_to_coordinates[idx] = landmark_px
  if connections:# 얼굴 선 연결
    num_landmarks = len(landmark_list.landmark)
    # Draws the connections if the start and end landmarks are both visible.
    # for connection in connections:
    #   start_idx = connection[0]
    #   end_idx = connection[1]
    #   if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):# 시작 인덱스와 끝 인덱스가 랜드마크 총 갯수 보다 작을 때 에러
    #     raise ValueError(f'Landmark index is out of range. Invalid connection '
    #                      f'from landmark #{start_idx} to landmark #{end_idx}.')
    #   if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
    #     cv2.line(image, idx_to_coordinates[start_idx],
    #              idx_to_coordinates[end_idx], connection_drawing_spec.color,
    #              connection_drawing_spec.thickness)
      #   print("index: {start_idx}".format(start_idx=start_idx))
      #   cv2.putText(image,"({0},{1})".format(start_idx,end_idx),org=(idx_to_coordinates[start_idx][0]-2,idx_to_coordinates[start_idx][1]-3),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.3,color=(255,255,255))
       #점찍기
      # if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
      #   cv2.circle(image, idx_to_coordinates[start_idx],2,(0,0,255))
      #   cv2.circle(image, idx_to_coordinates[end_idx], 2, (0, 0, 255))
      #   cv2.putText(image, "({0})".format(start_idx),
      #               org=(idx_to_coordinates[start_idx][0] - 2, idx_to_coordinates[start_idx][1] - 3),
      #               fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0, 0, 255))
      #   cv2.putText(image, "({0})".format(end_idx),
      #               org=(idx_to_coordinates[end_idx][0] - 2, idx_to_coordinates[end_idx][1] - 3),
      #               fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0, 0, 255))
    # 전체 점찍기
    try:
      success_flag = True
      for i in POINT:
        cv2.circle(image, (idx_to_coordinates[i][0],idx_to_coordinates[i][1]), 2, (0, 0, 255))
      #   cv2.putText(image, "({0})".format(i),
      #               org=(idx_to_coordinates[i][0] - 2, idx_to_coordinates[i][1] - 3),
      #               fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0, 0, 255))
      #   print("x: {0}, y: {1}".format(idx_to_coordinates[i][0],idx_to_coordinates[i][1]))
        #print(math.sqrt(math.pow((idx_to_coordinates[i][0]-idx_to_coordinates[i+1][0]),2)+math.pow((idx_to_coordinates[i][1]-idx_to_coordinates[i+1][1]),2)))#두점 거리
        #print(f"좌표 : {idx_to_coordinates[i]}")
        point_to_point_distance(idx_to_coordinates, image)#############특징점 잡는거!!!!!!!
      # cv2.imshow('hi', image)
      # cv2.waitKey(0)
    except Exception as e:
      print("오류 키", e)
      success_flag = False
      return idx_to_coordinates,success_flag

    return idx_to_coordinates,success_flag
  # Draws landmark points after finishing the connection lines, which is
  # aesthetically better.
  '''
  바꾼부분
  
  '''
  # file_list = {"iu1.jfif": 1}
  # for idx, file in enumerate(file_list):
  #   # print(file)
  #   image = cv2.imread(file)
  #   # Convert the BGR image to RGB before processing.
  #   # results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  #   # print("result: {0}".format(results))
  #
  #   annotated_image = image.copy()
  #   annotated_image = cv2.circle(annotated_image, idx_to_coordinates.values(), 1, (0, 0, 255), -1)  ###############
  #   cv2.imshow('hi', annotated_image)
  #   cv2.waitKey(0)
  '''
  끝
  '''
  # for landmark_px in idx_to_coordinates.values():
  #   cv2.circle(image, landmark_px, landmark_drawing_spec.circle_radius,
  #              landmark_drawing_spec.color, landmark_drawing_spec.thickness)
  #   cv2.imshow('hi', image)
  #   cv2.waitKey(0)
  #
  # cv2.imshow('hi', image)
  # time.sleep(3)


# def draw_axis(
#     image: np.ndarray,
#     rotation: np.ndarray,
#     translation: np.ndarray,
#     focal_length: Tuple[float, float] = (1.0, 1.0),
#     principal_point: Tuple[float, float] = (0.0, 0.0),
#     axis_length: float = 0.1,
#     x_axis_drawing_spec: DrawingSpec = DrawingSpec(color=(0, 0, 255)),
#     y_axis_drawing_spec: DrawingSpec = DrawingSpec(color=(0, 128, 0)),
#     z_axis_drawing_spec: DrawingSpec = DrawingSpec(color=(255, 0, 0))):
#   """Draws the 3D axis on the image.
#
#   Args:
#     image: A three channel RGB image represented as numpy ndarray.
#     rotation: Rotation matrix from object to camera coordinate frame.
#     translation: Translation vector from object to camera coordinate frame.
#     focal_length: camera focal length along x and y directions.
#     principal_point: camera principal point in x and y.
#     axis_length: length of the axis in the drawing.
#     x_axis_drawing_spec: A DrawingSpec object that specifies the x axis
#       drawing settings such as color, line thickness.
#     y_axis_drawing_spec: A DrawingSpec object that specifies the y axis
#       drawing settings such as color, line thickness.
#     z_axis_drawing_spec: A DrawingSpec object that specifies the z axis
#       drawing settings such as color, line thickness.
#
#   Raises:
#     ValueError: If one of the followings:
#       a) If the input image is not three channel RGB.
#   """
#   if image.shape[2] != RGB_CHANNELS:
#     raise ValueError('Input image must contain three channel rgb data.')
#   image_rows, image_cols, _ = image.shape
#   # Create axis points in camera coordinate frame.
#   axis_world = np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
#   axis_cam = np.matmul(rotation, axis_length*axis_world.T).T + translation
#   x = axis_cam[..., 0]
#   y = axis_cam[..., 1]
#   z = axis_cam[..., 2]
#   # Project 3D points to NDC space.
#   fx, fy = focal_length
#   px, py = principal_point
#   x_ndc = -fx * x / z + px
#   y_ndc = -fy * y / z + py
#   # Convert from NDC space to image space.
#   x_im = np.int32((1 + x_ndc) * 0.5 * image_cols)
#   y_im = np.int32((1 - y_ndc) * 0.5 * image_rows)
#   # Draw xyz axis on the image.
#   origin = (x_im[0], y_im[0])
#   x_axis = (x_im[1], y_im[1])
#   y_axis = (x_im[2], y_im[2])
#   z_axis = (x_im[3], y_im[3])
#   image = cv2.arrowedLine(image, origin, x_axis, x_axis_drawing_spec.color,
#                           x_axis_drawing_spec.thickness)
#   image = cv2.arrowedLine(image, origin, y_axis, y_axis_drawing_spec.color,
#                           y_axis_drawing_spec.thickness)
#   image = cv2.arrowedLine(image, origin, z_axis, z_axis_drawing_spec.color,
#                           z_axis_drawing_spec.thickness)
