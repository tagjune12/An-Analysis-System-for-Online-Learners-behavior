import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


from absl.flags import FLAGS

from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto


import easydict

FLAGS = easydict.EasyDict(
    {
        'framework' : 'tf',
        'weights' : './checkpoints/yolov4-tiny-416',
        'size' : 416,
        'tiny' : True,
        'model' : 'yolov4',
        'iou' : 0.30,
        'score' : 0.70
    }
)

class PeopleDetector:
    def __init__(self):
        self.input_size = FLAGS.size
        self.saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        self.infer = self.saved_model_loaded.signatures['serving_default']

    def detect(self, frame) -> (bool, list, list) :
        input_size = self.input_size
        infer = self.infer

        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        # if FLAGS.framework == 'tflite':
        #     interpreter.set_tensor(input_details[0]['index'], image_data)
        #     interpreter.invoke()
        #     pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        #     if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
        #         boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
        #                                         input_shape=tf.constant([input_size, input_size]))
        #     else:
        #         boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
        #                                         input_shape=tf.constant([input_size, input_size]))
        # else:
        #     batch_data = tf.constant(image_data)
        #     pred_bbox = infer(batch_data)  # 예측(모델 실행)하는 부분
        #     for key, value in pred_bbox.items():
        #         boxes = value[:, :, 0:4]
        #         pred_conf = value[:, :, 4:]

        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)  # 예측(모델 실행)하는 부분
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        image_start = np.zeros((1, 2), int)
        image_end = np.zeros((1, 2), int)
        # -------------------------------------------------------------------------------------------------------------------
        for i in range(valid_detections[0]):
            if int(classes[0][i]) < 0 or int(classes[0][i]) != 0: continue  # 검출된 객체가 사람이 아닐경우 반복문 끝으로

            coor = np.array(boxes[0][i])
            # Y좌표
            coor[0] = int(coor[0] * frame.shape[0])
            coor[2] = int(coor[2] * frame.shape[0])
            # X좌표
            coor[1] = int(coor[1] * frame.shape[1])
            coor[3] = int(coor[3] * frame.shape[1])

            image_start = np.concatenate((image_start, ([[int(coor[1]), int(coor[0])]])), axis=0)  # [x,y]
            image_end = np.concatenate((image_end, ([[int(coor[3]), int(coor[2])]])), axis=0)  # [x,y]

        image_start = np.delete(image_start, 0, axis=0)  # 검출된 사람의 영역 시작, 0으로 입력되어 있는 부분 삭제
        image_end = np.delete(image_end, 0, axis=0)  # 검출된 사람의 영역 끝, 0으로 입력되어 있는 부분 삭제

        return True, image_start, image_end





