import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
import joblib
import imutils
import argparse

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
    """
    Convert tensor to dictionary.

    Parameters:
        tensor (Tensor): Tensor containing key points of the body.

    Returns:
        dict: Dictionary containing key points of the body.
    """
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

if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument("--model-name", type=str, required=False, default='ymca_dance_pose_model',
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
    class_labels = {0: 'A', 1: 'C', 2: 'DANCE', 3: 'M', 4:'Y'}
    #class_labels = {0: 'Forward', 1: 'Left', 2: 'Right', 3: 'Stop'}

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

        except Exception as e:
            print(f"Error occurred: {e}")
            pass

        cv2.imshow('Pose Detection', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
