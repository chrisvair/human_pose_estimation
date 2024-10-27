import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
import joblib
import imutils
import argparse
import time
from time import sleep
from pylgbst import *
from pylgbst.hub import MoveHub
from pylgbst.peripherals import EncodedMotor, TiltSensor, Current, Voltage, COLORS, COLOR_BLACK
import logging

log = logging.getLogger("demo")

DEFAULT_IMAGE_WIDTH = 1200
X_TRANSLATION_PIXELS = 200
Z_TRANSLATION_PIXELS = 100

# Load a model
model = YOLO('yolov8n-pose.pt')

# Helpers for keypoints extraction:
BODY_PARTS = {"Nose": 0, "LEye": 1, "REye": 2, "LEar": 3, "REar": 4, "LShoulder": 5, "RShoulder": 6,
              "LElbow": 7,  "RElbow": 8, "LWrist": 9, "RWrist": 10, "LHip": 11, "RHip": 12, "LKnee": 13,
              "RKnee": 14,  "LAnkle": 15, "RAnkle": 16}

# Conversion of the predicted keypoints data to dictionary
def tensor_to_dict(tensor):
    key_points_dict = {}
    for part, i in BODY_PARTS.items():
        key_points_dict[part] = tensor[i].tolist()
    return key_points_dict

def reformat_keypoints(list_of_keypoints):
    reformated_list = []
    for points in list_of_keypoints:
        for coord in points:
            reformated_list.append(float(coord))
    return reformated_list

def turn_right(movehub, angle=200):
    log.info("Turning right")
    movehub.motor_AB.angled(angle, 0.5, -0.5)
    sleep(1)

def turn_left(movehub, angle=200):
    log.info("Turning left")
    movehub.motor_AB.angled(angle, -0.5, 0.5)
    sleep(1)

def stop(movehub):
    log.info("Stopping")
    movehub.motor_AB.stop()
    sleep(1)

def move_forward(movehub, duration=1):
    log.info("Moving forward")
    movehub.motor_AB.timed(duration, 0.5, 0.5)
    sleep(duration)

def get_options():
    import argparse
    arg_parser = argparse.ArgumentParser(
        description='Demonstrate move-hub communications',
    )
    arg_parser.add_argument(
        '-c', '--connection',
        default='auto://',
        help='''Specify connection URL to use, `protocol://mac?param=X` with protocol in:
    "gatt","pygatt","gattlib","gattool", "bluepy","bluegiga"'''
    )
    return arg_parser

def connection_from_url(url):
    import pylgbst
    if url == 'auto://':
        return None
    try:
        from urllib.parse import urlparse, parse_qs
    except ImportError:
        from urlparse import urlparse, parse_qs
    parsed = urlparse(url)
    name = 'get_connection_%s' % parsed.scheme
    factory = getattr(pylgbst, name, None)
    if not factory:
        msg = "Unrecognised URL scheme/protocol, expect a get_connection_<protocol> in pylgbst: %s"
        raise ValueError(msg % parsed.protocol)
    params = {}
    if parsed.netloc.strip():
        params['hub_mac'] = parsed.netloc
    for key, value in parse_qs(parsed.query).items():
        if len(value) == 1:
            params[key] = value[0]
        else:
            params[key] = value
    return factory(
        **params
    )

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(relativeCreated)d\t%(levelname)s\t%(name)s\t%(message)s')
    parser = get_options()
    options = parser.parse_args()
    parameters = {}
    try:
        connection = connection_from_url(options.connection)  # get_connection_bleak(hub_name=MoveHub.DEFAULT_NAME)
        parameters['connection'] = connection
    except ValueError as err:
        parser.error(err.args[0])

    hub = MoveHub(**parameters)

    try:
        ap = argparse.ArgumentParser()

        ap.add_argument("--model-name", type=str, required=False, default='robot_model_mac',
                        help="name of the saved pickled model [no suffix]")
        ap.add_argument("--suppress-landmarks", action='store_true',
                        help="[Optional: False] if present do not show landmarks on yourself ")
        ap.add_argument("--image-width", type=int, required=False, default=1200,
                        help="Image width")

        args = vars(ap.parse_args())
        DEFAULT_IMAGE_WIDTH = args['image_width']

        model_name = args['model_name']
        suppress_landmarks = args['suppress_landmarks']

        # Define the mapping from numerical labels to class names
        #class_labels = {0: 'A', 1: 'C', 2: 'M', 3: 'Y'}
        class_labels = {0: 'Forward', 1: 'Left', 2: 'Right', 3: 'Stop'}

        with open(f'{model_name}.pkl', 'rb') as f:
            prediction_model = joblib.load(f)
            print(prediction_model)

        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, frame = cap.read()
            frame = imutils.resize(frame, width=DEFAULT_IMAGE_WIDTH)

            # Run YOLOv8 inference on the frame
            results = model(frame, save=False)
            # Visualize the results on the frame
            frame = results[0].plot()

            try:
                for result in results:
                    for idx, keypoints_tensor in enumerate(result.keypoints.xyn):
                        key_points = tensor_to_dict(keypoints_tensor)
                arm_landmarks = [key_points['Nose'], key_points['LShoulder'], key_points['RShoulder'], key_points['LElbow'], key_points['RElbow'], key_points['LWrist'], key_points['RWrist']]
                arm_landmarks = reformat_keypoints(arm_landmarks)
                row = np.around(arm_landmarks, decimals=9).tolist()

                # Make Detections
                X = pd.DataFrame([row])
                body_language_class = prediction_model.predict(X)[0]
                body_language_prob = prediction_model.predict_proba(X)[0]
                print(body_language_class, np.around(body_language_prob, decimals=3))

                # Map the numerical class to the corresponding label
                body_language_class_str = class_labels.get(body_language_class, "Unknown")

                # Get status box
                status_width = 250
                cv2.rectangle(frame, (0, 0), (status_width, 60), (255, 182, 193), -1)

                # Display Class
                cv2.putText(frame, 'CLASS'
                            , (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, body_language_class_str
                            , (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Display Probability
                cv2.putText(frame, 'PROB'
                            , (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, str(round(body_language_prob[np.argmax(body_language_prob)], 2))
                            , (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Control the robot based on detected pose
                if body_language_prob[body_language_class] > 0.9:
                    if body_language_class_str == 'Forward':
                        move_forward(hub)
                    elif body_language_class_str == 'Right':
                        turn_right(hub)
                    elif body_language_class_str == 'Left':
                        turn_left(hub)
                    elif body_language_class_str == 'Stop':
                        stop(hub)

            except Exception as e:
                print(f"Error occurred: {e}")
                pass

            cv2.imshow('Pose Detection', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    finally:
        hub.disconnect()
