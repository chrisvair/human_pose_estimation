from ultralytics import YOLO
import cv2 # OpenCV for video capture
import argparse # parse command-line arguments
import csv
import numpy as np
from pathlib import Path
import time
from uielements.uielements import DisplayValueLabel

# Load a model
model = YOLO('yolov8n-pose.pt')

# Helpers for keypoints extraction:
BODY_PARTS = {"Nose": 0, "LEye": 1, "REye": 2, "LEar": 3, "REar": 4, "LShoulder": 5, "RShoulder": 6,
              "LElbow": 7,  "RElbow": 8, "LWrist": 9, "RWrist": 10, "LHip": 11, "RHip": 12, "LKnee": 13,
              "RKnee": 14,  "LAnkle": 15, "RAnkle": 16}
# Convertion of the predicted keypoints data to dictionary
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

# Define global variables for video capture:
_cap = None

# Define functions for video capture:
def _release_video():
    _cap.release()


def _initialize_video_capture():
    global _cap
    _cap = cv2.VideoCapture(0) # 0 for default camera


if __name__ == '__main__':
    # Command-line arguments:
    # in terminal: python3 data_collection.py --help
    ap = argparse.ArgumentParser()
    # ex in terminal: python3 data_collection.py --class-name Y --file-name YMCA_data.csv
    ap.add_argument("--class-name", required=True, help="target name for captured data")
    ap.add_argument("--collect-for", type=int, required=False, default=30, help="number of seconds to collect data")
    ap.add_argument("--start-delay", type=int, required=False, default=5,
                    help="number of seconds to wait before collecting data")
    ap.add_argument("--file-name", type=str, required=False, default='pose_training.csv',
                    help="name of the training data file")
    ap.add_argument("--dry-run", action='store_true', help="[Optional: False] is set then do NOT store data")
    args = vars(ap.parse_args())

    class_name = args['class_name']
    collect_for = args['collect_for']
    start_delay = args['start_delay']
    file_name = args['file_name']
    dry_run = args['dry_run']

    _initialize_video_capture()

    # Initiate holistic model
    start_delay_time = time.time() + start_delay
    collect_for_time = start_delay_time + collect_for

    display_label = DisplayValueLabel(3,3, 200, 40, "Start in")
    print(f"***** Start capture in {start_delay} seconds ")

    # save an image of the pose so we can overlay points
    saved_image = False
    while _cap.isOpened():
        success, frame = _cap.read()
        if success:
            # Wait 'start_delay_time' before starting to collect data points to give the person
            # time to get into position
            if time.time() > start_delay_time:
                # have we saved an initial image yet?
                if not saved_image:
                    if not dry_run:
                        print(f"Image Shape: {frame.shape}")
                        cv2.imwrite(f'../data/{class_name}.png', frame)
                        saved_image = True
                display_label.label = 'Seconds'
                    
                # Run YOLOv8 inference on the frame
                results = model(frame, save=False)
                # Visualize the results on the frame
                frame = results[0].plot()
            
                # display remaining collection time
                display_label.set_value(int(collect_for_time-time.time()))
                display_label.draw(frame)
                
                # Display the annotated frame
                cv2.imshow("YOLOv8 Inference", frame)
                
                # Export coordinates
                try:
                    # Iterate through the detected keypoints
                    count = 0
                    for result in results:
                        for idx, keypoints_tesor in enumerate(result.keypoints.xyn):
                            key_points = tensor_to_dict(keypoints_tesor)
                    arm_landmarks = [key_points['Nose'], key_points['LShoulder'], key_points['RShoulder'], key_points['LElbow'], key_points['RElbow'], key_points['LWrist'], key_points['RWrist']]
                    arm_landmarks = reformat_keypoints(arm_landmarks)
                    row = np.around(arm_landmarks, decimals=9).tolist()
                    # Append class name
                    row.insert(0, class_name)
                    # Export to CSV
                    if not dry_run:
                        Path(f'../data').mkdir(parents=True, exist_ok=True)
                        with open(f'../data/{file_name}', mode='a', newline='') as f:
                            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            csv_writer.writerow(row)
                        
                except Exception as e:
                    print(f"Error occurred: {e}")
                    pass

            else:
                display_label.set_value(int(start_delay_time-time.time()))
                display_label.draw(frame)

            cv2.imshow('Pose Detection', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            if time.time() > collect_for_time:
                break

    _release_video()

    cv2.destroyAllWindows()