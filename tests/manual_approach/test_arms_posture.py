import unittest
import torch # create tensors
import sys
import numpy as np
sys.path.append('../src/manual_approach')
from src.manual_approach.arms_posture import *

class TestArmsPostures(unittest.TestCase):

    def test_tensor_to_dict(self):
        tensor = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8],
                       [9, 9], [10, 10], [11, 11], [12, 12], [13, 13], [14, 14], [15, 15], [16, 16]])
        expected_result = {"Nose": [0, 0], "LEye": [1, 1], "REye": [2, 2], "LEar": [3, 3], "REar": [4, 4], "LShoulder": [5, 5], "RShoulder": [6, 6],
              "LElbow": [7, 7],  "RElbow": [8, 8], "LWrist": [9, 9], "RWrist": [10, 10], "LHip": [11, 11], "RHip": [12, 12], "LKnee": [13, 13],
              "RKnee": [14, 14],  "LAnkle": [15, 15], "RAnkle": [16, 16]}
        result = tensor_to_dict(tensor)
        try:
            self.assertDictEqual(result, expected_result)
        except AssertionError:
            print("Result:", result)
            print("Expected result:", expected_result)
            raise

    def test_calculate_angle_cos(self):
        p1 = [1, 1]
        p2 = [2, 2]
        p3 = [3, 3]
        result = calculate_angle_cos(p1, p2, p3)
        expected_result = 1
        try:
            self.assertAlmostEqual(result, expected_result)
        except AssertionError:
            print("Result:", result)
            print("Expected result:", expected_result)
            raise

    def test_calculate_angle(self):
        p1 = [1, 1]
        p2 = [2, 2]
        p3 = [3, 3]
        result = calculate_angle(p1, p2, p3)
        expected_result = 180
        try:
            self.assertTrue(np.isclose(result, expected_result, atol=1e-5))
        except AssertionError:
            print("Result:", result)
            print("Expected result:", expected_result)
            raise
    
    def test_left_arm_straight(self):
        key_points1 = {
            'LShoulder': [0.1, 0.1],
            'LElbow': [0.2, 0.2],
            'LWrist': [0.3, 0.3]
        }
        result = left_arm_straight(key_points1)
        self.assertTrue(result)
        key_points2 = {
            'LShoulder': [0.1, 0.1],
            'LElbow': [0.6, 0.6],
            'LWrist': [0.3, 0.3]
        }
        result = left_arm_straight(key_points2)
        self.assertFalse(result)
    
    def test_right_arm_straight(self):
        key_points = {
            'RShoulder': [0.3, 0.3],
            'RElbow': [0.2, 0.2],
            'RWrist': [0.1, 0.1]
        }
        result = right_arm_straight(key_points)
        self.assertTrue(result)
    
    def test_arms_straight(self):
        key_points = {
            'RShoulder': [0.3, 0.3],
            'RElbow': [0.2, 0.2],
            'RWrist': [0.1, 0.1],
            'LShoulder': [0.4, 0.4],
            'LElbow': [0.5, 0.5],
            'LWrist': [0.6, 0.6]
        }
        result = arms_straight(key_points)
        self.assertTrue(result)
    
    def test_left_arm_overhead(self):
        key_points1 = {
            'Nose': [0.1, 0.5],
            'LElbow': [0.2, 0.2],
            'LWrist': [0.3, 0.1]
        }
        result = left_arm_overhead(key_points1)
        self.assertTrue(result)

        key_points2 = {
            'Nose': [0.1, 0.5],
            'LElbow': [0.2, 0.6],
            'LWrist': [0.3, 0.1]
        }
        result = left_arm_overhead(key_points2)
        self.assertFalse(result)

    def test_right_arm_overhead(self):
        key_points = {
            'Nose': [0.1, 0.5],
            'RElbow': [0.2, 0.2],
            'RWrist': [0.3, 0.1]
        }
        result = right_arm_overhead(key_points)
        self.assertTrue(result)

    def test_arms_overhead(self):
        key_points = {
            'Nose': [0.1, 0.5],
            'RElbow': [0.2, 0.2],
            'RWrist': [0.3, 0.1],
            'LElbow': [0.3, 0.2],
            'LWrist': [0.4, 0.1]
        }
        result = arms_overhead(key_points)
        self.assertTrue(result)
    
    def test_left_arm_outward(self):
        key_points1 = {
            'Nose': [0.5, 0.5],
            'LElbow': [0.6, 0.5],
            'LWrist': [0.7, 0.6]
        }
        result = left_arm_outward(key_points1)
        self.assertTrue(result)

        key_points2 = {
            'Nose': [0.5, 0.5],
            'LElbow': [0.4, 0.5],
            'LWrist': [0.3, 0.6]
        }
        result = left_arm_outward(key_points2)
        self.assertFalse(result)

    def test_right_arm_outward(self):
        key_points = {
            'Nose': [0.5, 0.5],
            'RElbow': [0.4, 0.5],
            'RWrist': [0.3, 0.6]
        }
        result = right_arm_outward(key_points)
        self.assertTrue(result)

    def test_arms_outward(self):
        key_points = {
            'Nose': [0.5, 0.5],
            'RElbow': [0.4, 0.5],
            'RWrist': [0.3, 0.6],
            'LElbow': [0.6, 0.5],
            'LWrist': [0.7, 0.6]
        }
        result = arms_outward(key_points)
        self.assertTrue(result)
    
    def test_left_arm_right_angle(self):
        key_points1 = {
            'LShoulder': [0.1, 0.1],
            'LElbow': [0.2, 0.2],
            'LWrist': [0.3, 0.1]
        }
        result = left_arm_right_angle(key_points1)
        self.assertTrue(result)

        key_points2 = {
            'LShoulder': [0.1, 0.1],
            'LElbow': [0.2, 0.2],
            'LWrist': [0.3, 0.3]
        }
        result = left_arm_right_angle(key_points2)
        self.assertFalse(result)

    def test_right_arm_right_angle(self):
        key_points = {
            'RShoulder': [0.1, 0.1],
            'RElbow': [0.2, 0.2],
            'RWrist': [0.3, 0.1]
        }
        result = right_arm_right_angle(key_points)
        self.assertTrue(result)

    def test_arms_right_angle(self):
        key_points = {
            'RShoulder': [0.0, 0.1],
            'RElbow': [0.1, 0.2],
            'RWrist': [0.2, 0.1],
            'LShoulder': [0.1, 0.1],
            'LElbow': [0.2, 0.2],
            'LWrist': [0.3, 0.1]
        }
        result = arms_right_angle(key_points)
        self.assertTrue(result)
    
    def test_left_wrist_overhead(self):
        key_points1 = {
            'Nose': [0.5, 0.3],
            'LWrist': [0.5, 0.5]
        }
        result = left_wrist_overhead(key_points1)
        self.assertFalse(result)

        key_points2 = {
            'Nose': [0.5, 0.7],
            'LWrist': [0.5, 0.5]
        }
        result = left_wrist_overhead(key_points2)
        self.assertTrue(result)

    def test_right_wrist_overhead(self):
        key_points1 = {
            'Nose': [0.5, 0.3],
            'RWrist': [0.5, 0.5]
        }
        result = right_wrist_overhead(key_points1)
        self.assertFalse(result)

        key_points2 = {
            'Nose': [0.5, 0.7],
            'RWrist': [0.5, 0.5]
        }
        result = right_wrist_overhead(key_points2)
        self.assertTrue(result)

    def test_wrists_overhead(self):
        key_points = {
            'Nose': [0.5, 0.5],
            'LWrist': [0.5, 0.3],
            'RWrist': [0.4, 0.3]
        }
        result = wrists_overhead(key_points)
        self.assertTrue(result)
    
    def test_left_arm_horizontal(self):
        key_points1 = {
            'Neck': [0.5, 0.5],
            'LShoulder': [0.4, 0.6],
            'LElbow': [0.3, 0.7],
            'LWrist': [0.2, 0.8]
        }
        result = left_arm_horizontal(key_points1)
        self.assertTrue(result)

        key_points2 = {
            'Neck': [0.5, 0.5],
            'LShoulder': [0.4, 0.6],
            'LElbow': [0.3, 0.7],
            'LWrist': [0.3, 0.8]
        }
        result = left_arm_horizontal(key_points2)
        self.assertFalse(result)

    def test_right_arm_horizontal(self):
        key_points1 = {
            'Neck': [0.5, 0.5],
            'RShoulder': [0.6, 0.6],
            'RElbow': [0.7, 0.7],
            'RWrist': [0.8, 0.8]
        }
        result = right_arm_horizontal(key_points1)
        self.assertTrue(result)

        key_points2 = {
            'Neck': [0.5, 0.5],
            'RShoulder': [0.6, 0.6],
            'RElbow': [0.7, 0.7],
            'RWrist': [0.6, 0.8]
        }
        result = right_arm_horizontal(key_points2)
        self.assertFalse(result)

    def test_arms_horizontal(self):
        key_points = {
            'Neck': [0.5, 0.5],
            'LShoulder': [0.4, 0.6],
            'LElbow': [0.3, 0.7],
            'LWrist': [0.2, 0.8],
            'RShoulder': [0.6, 0.6],
            'RElbow': [0.7, 0.7],
            'RWrist': [0.8, 0.8]
        }
        result = arms_horizontal(key_points)
        self.assertTrue(result)
    def test_right_arm_underhip(self):
        key_points1 = {
            'RHip': [0.5, 0.5],
            'RWrist': [0.5, 0.7]
        }
        result = right_arm_underhip(key_points1)
        self.assertTrue(result)

        key_points2 = {
            'RHip': [0.5, 0.5],
            'RWrist': [0.5, 0.4]
        }
        result = right_arm_underhip(key_points2)
        self.assertFalse(result)

    def test_left_arm_underhip(self):
        key_points1 = {
            'LHip': [0.5, 0.5],
            'LWrist': [0.5, 0.7]
        }
        result = left_arm_underhip(key_points1)
        self.assertTrue(result)

        key_points2 = {
            'LHip': [0.5, 0.5],
            'LWrist': [0.5, 0.4]
        }
        result = left_arm_underhip(key_points2)
        self.assertFalse(result)

    def test_arms_underhip(self):
        key_points = {
            'RHip': [0.5, 0.5],
            'RWrist': [0.5, 0.7],
            'LHip': [0.5, 0.5],
            'LWrist': [0.5, 0.7]
        }
        result = arms_underhip(key_points)
        self.assertTrue(result)
    
    def test_left_arm_angle(self):
        key_points1 = {
            'LShoulder': [0.0, 0.0],
            'LElbow': [0.5, 0.5],
            'LWrist': [1.0, 0.0]
        }
        result = left_arm_angle(key_points1, 45, 135, 10)
        self.assertTrue(result)

        key_points2 = {
            'LShoulder': [0.0, 0.0],
            'LElbow': [0.5, 0.5],
            'LWrist': [1.0, 0.0]
        }
        result = left_arm_angle(key_points2, 135, 180, 10)
        self.assertFalse(result)

    def test_right_arm_angle(self):
        key_points1 = {
            'RShoulder': [1.0, 0.0],
            'RElbow': [0.5, 0.5],
            'RWrist': [0.0, 0.0]
        }
        result = right_arm_angle(key_points1, 45, 135, 10)
        self.assertTrue(result)

        key_points2 = {
            'RShoulder': [1.0, 0.0],
            'RElbow': [0.5, 0.5],
            'RWrist': [0.0, 0.0]
        }
        result = right_arm_angle(key_points2, 135, 180, 10)
        self.assertFalse(result)

    def test_arms_angle(self):
        key_points = {
            'LShoulder': [0.0, 0.0],
            'LElbow': [0.5, 0.5],
            'LWrist': [1.0, 0.0],
            'RShoulder': [1.0, 0.0],
            'RElbow': [0.5, 0.5],
            'RWrist': [0.0, 0.0]
        }
        result = arms_angle(key_points, 45, 135, 10)
        self.assertTrue(result)

    def test_hands_close(self):
        key_points1 = {
            'LWrist': [0.0, 0.0],
            'RWrist': [0.0, 0.2]
        }
        result = hands_close(key_points1, 0.3)
        self.assertTrue(result)

        key_points2 = {
            'LWrist': [0.0, 0.0],
            'RWrist': [0.0, 0.4]
        }
        result = hands_close(key_points2, 0.3)
        self.assertFalse(result)
        
if __name__ == '__main__':
    unittest.main()