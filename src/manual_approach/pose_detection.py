from ultralytics import YOLO

# Load a model
model = YOLO('./yolov8x-pose-p6.pt')  # load the pretrained model
import numpy as np

path = "../entries/picture_originals/"
file_name = 'NAME_OF_YOUR_PICTURE.png'
pose = file_name[0]
source = path+''+file_name
save = model.predict(source, save = True, conf =0.5, project="../pictures/picture_results", name=file_name)  # predict on an image
results = model(source)


csv_file_path = './dataset/keypoints_dataset.csv'
from arms_posture import (
    left_arm_overhead,
    left_arm_right_angle,
    right_arm_right_angle,
    arms_right_angle,
    left_wrist_overhead,
    right_wrist_overhead,
    wrists_overhead,
    left_arm_horizontal,
    right_arm_horizontal,
    left_arm_angle,
    right_arm_angle,
    arms_angle,
    wrists_close,
    tensor_to_dict,
    hands_angles_similar,
    wrists_over_shoulders
)

## YMCA poses

def is_Y(key_points):
    '''Detects the Y pose'''
    return wrists_overhead(key_points) and arms_angle(key_points, 140, 180) and not wrists_close(key_points)
def is_A(key_points):
    '''Detects the A pose'''
    return wrists_overhead(key_points) and wrists_close(key_points)
def is_M(key_points):
    '''Detects the M pose'''
    return arms_angle(key_points, 20, 80) and wrists_close(key_points) and hands_angles_similar(key_points) and wrists_over_shoulders(key_points)
def is_C(key_points):
    '''Detects the C pose'''
    # return left_wrist_overhead(key_points) and not right_wrist_overhead(key_points) and left_arm_angle(key_points, 70, 140) and right_arm_angle(key_points, 70, 160)
    return left_wrist_overhead(key_points) and not right_wrist_overhead(key_points)
def is_Dance(key_points):
    '''Detects the Dance pose'''
    return not wrists_overhead(key_points)

count = 0
for result in results:
    for idx, person in enumerate(result.keypoints.xyn):
        confidence = result.boxes.conf[idx]
        # confidence threshold 0.5
        if confidence > 0.5:
            count += 1
            print("Person", count, person)
            print(person)
            key_points = tensor_to_dict(person)
            print(key_points)
            print('Is it pose Y?', is_Y(key_points))
            print('Is it pose A?', is_A(key_points))
            print('Is it pose M?', is_M(key_points))
            print('Is it pose C?', is_C(key_points))