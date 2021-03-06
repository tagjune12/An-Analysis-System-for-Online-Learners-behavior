import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
# from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import mediapipe as mp

from PIL import ImageGrab

import time

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/road.mp4', 'path to input video')
flags.DEFINE_float('iou', 0.30, 'iou threshold')
flags.DEFINE_float('score', 0.20, 'score threshold')
flags.DEFINE_string('output', None, 'path to output video')
# flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_string('output_format', None, 'codec used in VideoWriter when saving video to file')
flags.DEFINE_boolean('dis_cv2_window', False, 'disable cv2 window during the process')  # this is good for the .ipynb


def main(_argv):
    from tensorflow.python.client import device_lib
    device_lib.list_local_devices()

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    # session = InteractiveSession(config=config)
    # STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    # video_path = FLAGS.video

    # print("Video from: ", video_path)
    # vid = cv2.VideoCapture(video_path)
    vid = cv2.VideoCapture(0)

    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_id = 0
    '''
    while True:
        # start_time = time.time()
        # return_value, frame = vid.read()
        return_value = True
        image = ImageGrab.grab(bbox=(50, 200, 1000, 1000))  # ???????????? ???????????? ????????? ??????##@!#!@
        frame = np.array(image)  # ???????????? ????????? ??????##@!#!@#!@
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        else:
            if frame_id == vid.get(cv2.CAP_PROP_FRAME_COUNT):
                print("Video processing complete")
                break
            raise ValueError("No image! Try with another video format")

        # frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)  # ??????(?????? ??????)?????? ??????
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
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

        image_start = np.zeros((1, 2), int)
        image_end = np.zeros((1, 2), int)
        # -------------------------------------------------------------------------------------------------------------------
        mp_drawing = mp.solutions.drawing_utils
        mp_holistic = mp.solutions.holistic

        for i in range(valid_detections[0]):
            # if int(classes[0][i]) < 0 or int(classes[0][i]) != 0: continue # ????????? ????????? ????????? ???????????? ????????? ?????????
            print(classes[0])
            if int(classes[0][i]) < 77: continue  # ????????? ????????? ??????, ???, ?????? ???????????? ????????? ?????????

            coor = np.array(boxes[0][i])
            # Y??????
            coor[0] = int(coor[0] * frame.shape[0])
            coor[2] = int(coor[2] * frame.shape[0])
            # X??????
            coor[1] = int(coor[1] * frame.shape[1])
            coor[3] = int(coor[3] * frame.shape[1])

            image_start = np.concatenate((image_start, ([[int(coor[1]), int(coor[0])]])), axis=0)  # [x,y]
            image_end = np.concatenate((image_end, ([[int(coor[3]), int(coor[2])]])), axis=0)  # [x,y]

        image_start = np.delete(image_start, 0, axis=0)  # ????????? ????????? ?????? ??????, 0?????? ???????????? ?????? ?????? ??????
        image_end = np.delete(image_end, 0, axis=0)  # ????????? ????????? ?????? ???, 0?????? ???????????? ?????? ?????? ??????

        for i in range(len(image_start)):
            start = tuple(image_start[i])
            end = tuple(image_end[i])
            # image -> human ????????? crop ??? ?????????
            image = frame[start[1]:end[1], start[0]:end[0]]  # ?????????[Y??????, X??????]
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            # with mp_holistic.Holistic(static_image_mode=True) as holistic:
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                annotated_image = image.copy()  # ?????? ?????? ?????????
                # mp_drawing.draw_landmarks(
                # #     annotated_image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
                # mp_drawing.draw_landmarks(
                #     annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                # frame -> ???????????? ???????????? ?????? ?????????
                # ??? ?????? ????????? ???????????? ????????? ???????????? ????????? ??????
                frame[start[1]:end[1], start[0]:end[0]] = annotated_image
        # ----------------------------------------------------------------------------------------------------------------
        image = utils.draw_bbox(frame, pred_bbox)  # ???????????? ????????? ????????? ???????????? ?????? ????????? ????????? ????????? ???
        image = cv2.resize(image, (1100, 640))  # 1280,720 ????????? ????????? ??????
        # result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        result = image
        end_time = time.time()
        # print("FPS: ",  frame_id / int(end_time-start_time))
        if not FLAGS.dis_cv2_window:
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        if FLAGS.output:
            out.write(result)

        frame_id += 1
    '''
    while True:
        try:
            # start_time = time.time()
            # return_value, frame = vid.read()
            return_value = True
            image = ImageGrab.grab(bbox=(50, 200, 1000, 1000))  # ???????????? ???????????? ????????? ??????##@!#!@
            frame = np.array(image)  # ???????????? ????????? ??????##@!#!@#!@
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            else:
                if frame_id == vid.get(cv2.CAP_PROP_FRAME_COUNT):
                    print("Video processing complete")
                    break
                raise ValueError("No image! Try with another video format")

            # frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)

            if FLAGS.framework == 'tflite':
                interpreter.set_tensor(input_details[0]['index'], image_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
            else:
                batch_data = tf.constant(image_data)
                pred_bbox = infer(batch_data)  # ??????(?????? ??????)?????? ??????
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
            pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

            image_start = np.zeros((1, 2), int)
            image_end = np.zeros((1, 2), int)
            # -------------------------------------------------------------------------------------------------------------------
            mp_drawing = mp.solutions.drawing_utils
            mp_holistic = mp.solutions.holistic

            for i in range(valid_detections[0]):
                # if int(classes[0][i]) < 0 or int(classes[0][i]) != 0: continue # ????????? ????????? ????????? ???????????? ????????? ?????????
                print(classes[0])
                if int(classes[0][i]) < 77: continue  # ????????? ????????? ??????, ???, ?????? ???????????? ????????? ?????????

                coor = np.array(boxes[0][i])
                # Y??????
                coor[0] = int(coor[0] * frame.shape[0])
                coor[2] = int(coor[2] * frame.shape[0])
                # X??????
                coor[1] = int(coor[1] * frame.shape[1])
                coor[3] = int(coor[3] * frame.shape[1])

                image_start = np.concatenate((image_start, ([[int(coor[1]), int(coor[0])]])), axis=0)  # [x,y]
                image_end = np.concatenate((image_end, ([[int(coor[3]), int(coor[2])]])), axis=0)  # [x,y]

            image_start = np.delete(image_start, 0, axis=0)  # ????????? ????????? ?????? ??????, 0?????? ???????????? ?????? ?????? ??????
            image_end = np.delete(image_end, 0, axis=0)  # ????????? ????????? ?????? ???, 0?????? ???????????? ?????? ?????? ??????

            for i in range(len(image_start)):
                start = tuple(image_start[i])
                end = tuple(image_end[i])
                # image -> human ????????? crop ??? ?????????
                image = frame[start[1]:end[1], start[0]:end[0]]  # ?????????[Y??????, X??????]
                image_height, image_width, _ = image.shape
                # Convert the BGR image to RGB before processing.
                # with mp_holistic.Holistic(static_image_mode=True) as holistic:
                with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                    annotated_image = image.copy()  # ?????? ?????? ?????????
                    # mp_drawing.draw_landmarks(
                    # #     annotated_image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
                    # mp_drawing.draw_landmarks(
                    #     annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                    # frame -> ???????????? ???????????? ?????? ?????????
                    # ??? ?????? ????????? ???????????? ????????? ???????????? ????????? ??????
                    frame[start[1]:end[1], start[0]:end[0]] = annotated_image
            # ----------------------------------------------------------------------------------------------------------------
            image = utils.draw_bbox(frame, pred_bbox)  # ???????????? ????????? ????????? ???????????? ?????? ????????? ????????? ????????? ???
            image = cv2.resize(image, (1100, 640))  # 1280,720 ????????? ????????? ??????
            # result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            result = image
            end_time = time.time()
            # print("FPS: ",  frame_id / int(end_time-start_time))
            if not FLAGS.dis_cv2_window:
                cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
                cv2.imshow("result", result)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

            if FLAGS.output:
                out.write(result)

            frame_id += 1
        except Exception as e:
            print(e)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
