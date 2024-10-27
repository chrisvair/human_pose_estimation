import numpy as np

''' 
Posture Detection Functions Toolbox

These functions detect various postures based on key points of the body detected in an image.

Parameters: 
    key_points (dict): Dictionary containing key points of the body.
    threshold (float, optional): Threshold for cosine similarity. Defaults to tolerance.
    
Returns:
     bool: True if the posture is correct, False otherwise.
'''

BODY_PARTS = {"Nose": 0, "LEye": 1, "REye": 2, "LEar": 3, "REar": 4, "LShoulder": 5, "RShoulder": 6,
              "LElbow": 7,  "RElbow": 8, "LWrist": 9, "RWrist": 10, "LHip": 11, "RHip": 12, "LKnee": 13,
              "RKnee": 14,  "LAnkle": 15, "RAnkle": 16}

tolerance = 0.8

# Convertion of the predicted keypoints data to use them in the functions below
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


def calculate_angle_cos(p1, p2, p3):
    """
    Helper function to calculate the cosine of the angle between three points.
    To meet the threshold, we want to have an angle of 0 degrees for a straight line
    """
    # Calculate the vectors
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p2)
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return cos_angle

def calculate_angle(p1, p2, p3):
    """
    Helper function to calculate the angle in degrees between three points.
    """
    # Calculate the vectors
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    
    # Calculate the angle in radians
    angle_rad = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    # Convert to degrees
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def left_arm_straight(key_points, threshold=tolerance):
    if key_points['LShoulder'] and key_points['LElbow'] and key_points['LWrist']:
        cosinus = calculate_angle_cos(key_points['LShoulder'], key_points['LElbow'], key_points['LWrist'])
        return cosinus > threshold
    return False

def right_arm_straight(key_points, threshold=tolerance):
    if key_points['RShoulder'] and key_points['RElbow'] and key_points['RWrist']:
        cosinus = calculate_angle_cos(key_points['RShoulder'], key_points['RElbow'], key_points['RWrist'])
        return cosinus > threshold
    return False

def arms_straight(key_points):
    return right_arm_straight(key_points) and left_arm_straight(key_points)

def left_arm_overhead(key_points):
    if key_points['Nose'] and key_points['LElbow'] and key_points['LWrist']:
        return all(y < key_points['Nose'][1] for y in (key_points['LElbow'][1], key_points['LWrist'][1]))
    return False

def right_arm_overhead(key_points):
    if key_points['Nose'] and key_points['RElbow'] and key_points['RWrist']:
        return all(y < key_points['Nose'][1] for y in (key_points['RElbow'][1], key_points['RWrist'][1]))
    return False

def arms_overhead(key_points):
    return right_arm_overhead(key_points) and left_arm_overhead(key_points)

def left_arm_outward(key_points):
    if key_points['Nose'] and key_points['LElbow'] and key_points['LWrist']:
        return all([x > key_points['Nose'][0] for x in (key_points['LElbow'][0], key_points['LWrist'][0])])
    return False

def right_arm_outward(key_points):
    if key_points['Nose'] and key_points['RElbow'] and key_points['RWrist']:
        return all([x < key_points['Nose'][0] for x in (key_points['RElbow'][0], key_points['RWrist'][0])])
    return False

def arms_outward(key_points):
    return right_arm_outward(key_points) and left_arm_outward(key_points)

def left_arm_right_angle(key_points, threshold = 1-tolerance):
    if key_points['LShoulder'] and key_points['LElbow'] and key_points['LWrist']:
        cosinus = calculate_angle_cos(key_points['LShoulder'], key_points['LElbow'], key_points['LWrist'])
        return cosinus < threshold
    return False

def right_arm_right_angle(key_points, threshold = 1-tolerance):
    if key_points['RShoulder'] and key_points['RElbow'] and key_points['RWrist']:
        cosinus = calculate_angle_cos(key_points['RShoulder'], key_points['RElbow'], key_points['RWrist'])
        return cosinus < threshold
    return False

def arms_right_angle(key_points):
    return right_arm_right_angle(key_points) and left_arm_right_angle(key_points)

# Normalized coordinates -> converting the pixel coordinates (x, y) to values between 0 and 1,
# where (0, 0) corresponds to the top-left corner of the image, and (1, 1) corresponds to the bottom-right corner.
def left_wrist_overhead(key_points): 
    if key_points['Nose'] and key_points['LWrist']:
        return key_points['Nose'][1] > key_points['LWrist'][1]
    return False

def right_wrist_overhead(key_points):
    if key_points['Nose'] and key_points['RWrist']:
        return key_points['Nose'][1] > key_points['RWrist'][1]
    return False

def wrists_overhead(key_points):
    return right_wrist_overhead(key_points) and left_wrist_overhead(key_points)

def left_arm_horizontal(key_points, threshold=tolerance):
    if not left_arm_straight(key_points):
        return False
    if key_points['Neck']:
        cosinus = calculate_angle_cos(key_points['Neck'], key_points['LShoulder'], key_points['LElbow'])
        return cosinus > threshold
    return False

def right_arm_horizontal(key_points, threshold=tolerance):
    if not right_arm_straight(key_points):
        return False
    if key_points['Neck']:
        cosinus = calculate_angle_cos(key_points['Neck'], key_points['RShoulder'], key_points['RElbow'])
        return cosinus > threshold
    return False

def arms_horizontal(key_points):
    return right_arm_horizontal(key_points) and left_arm_horizontal(key_points)

def right_arm_underhip(key_points):
    if key_points['RHip'] and key_points['RWrist']:
        return key_points['RHip'][1] < key_points['RWrist'][1]
    return False

def left_arm_underhip(key_points):
    if key_points['LHip'] and key_points['LWrist']:
        return key_points['LHip'][1] < key_points['LWrist'][1]
    return False

def arms_underhip(key_points):
    return right_arm_underhip(key_points) and left_arm_underhip(key_points)

# Left arm forms an angle between min_angle and max_angle degrees
def left_arm_angle(key_points, min_angle, max_angle, tolerance=10):
    if key_points['LShoulder'] and key_points['LElbow'] and key_points['LWrist']:
        angle = calculate_angle(key_points['LShoulder'], key_points['LElbow'], key_points['LWrist'])
        return (min_angle - tolerance) <= angle <= (max_angle + tolerance)
    return False

# Right arm forms an angle between min_angle and max_angle degrees
def right_arm_angle(key_points, min_angle, max_angle, tolerance=10):
    if key_points['RShoulder'] and key_points['RElbow'] and key_points['RWrist']:
        angle = calculate_angle(key_points['RShoulder'], key_points['RElbow'], key_points['RWrist'])
        return (min_angle - tolerance) <= angle <= (max_angle + tolerance)
    return False

# Both arms form an angle between min_angle and max_angle degrees
def arms_angle(key_points, min_angle, max_angle, tolerance=10):
    return left_arm_angle(key_points, min_angle, max_angle, tolerance) and right_arm_angle(key_points, min_angle, max_angle, tolerance)


def wrists_close(key_points, threshold=0.2):
    if key_points['LWrist'] and key_points['RWrist']:
        left_wrist = np.array(key_points['LWrist'])
        right_wrist = np.array(key_points['RWrist'])
        distance = np.linalg.norm(left_wrist - right_wrist)
        return distance < threshold
    return False

def hands_angles_similar(key_points, tolerance=30):
    if key_points['LShoulder'] and key_points['LElbow'] and key_points['LWrist'] and key_points['RShoulder'] and key_points['RElbow'] and key_points['RWrist']:
        left_angle = calculate_angle(key_points['LShoulder'], key_points['LElbow'], key_points['LWrist'])
        right_angle = calculate_angle(key_points['RShoulder'], key_points['RElbow'], key_points['RWrist'])
        return abs(left_angle - right_angle) <= tolerance
    return False

def wrists_over_shoulders(key_points):
    if key_points['LShoulder'] and key_points['RShoulder'] and key_points['LWrist'] and key_points['RWrist']:
        return key_points['LWrist'][1] < key_points['LShoulder'][1] and key_points['RWrist'][1] < key_points['RShoulder'][1]
    return False


