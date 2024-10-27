import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from pose_detection import (
    is_Y,
    is_M,
    is_C,
    is_A,
    is_Dance
)

from arms_posture import BODY_PARTS

# Path to your CSV file
csv_file_path = '../data/pose_data.csv'

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Extract keypoints
X = df.iloc[:, 1:]
# Extract classes
y = df.iloc[:, 0]

# Lists to store true and predicted labels
true_labels = []
predicted_labels = []

def reformat(data):
    key_points_dict = {}
    for part, i in BODY_PARTS.items():
        if part == "Nose":
            key_points_dict[part] = [data.iloc[0], data.iloc[1]]
        elif part == "LShoulder":
            key_points_dict[part] = [data.iloc[2], data.iloc[3]]
        elif part == "RShoulder":
            key_points_dict[part] = [data.iloc[4], data.iloc[5]]
        elif part == "LElbow":
            key_points_dict[part] = [data.iloc[6], data.iloc[7]]
        elif part == "RElbow":
            key_points_dict[part] = [data.iloc[8], data.iloc[9]]
        elif part == "LWrist":
            key_points_dict[part] = [data.iloc[10], data.iloc[11]]
        elif part == "RWrist":
            key_points_dict[part] = [data.iloc[12], data.iloc[13]]
        else:
            key_points_dict[part] = [0, 0]
    return key_points_dict

# Iterate over the DataFrame rows
for index, row in df.iterrows():
    keypoints = row.iloc[1:]  # Assuming keypoints start from the second column
    true_label = row.iloc[0]  # Assuming the label is in the first column

    true_labels.append(true_label)

    keypoints = reformat(keypoints)
    if is_Y(keypoints):
        predicted_labels.append("Y")
    elif is_M(keypoints):
        predicted_labels.append("M")
    elif is_C(keypoints):
        predicted_labels.append("C")
    elif is_A(keypoints):
        predicted_labels.append("A")
    # elif is_Dance(keypoints):
    #     predicted_labels.append("DANCE-legs")
    else:
        predicted_labels.append("None")

print(true_labels)
print(predicted_labels)

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print("Accuracy:", accuracy)
# Accuracy: 0.8372280419016922

# Calculate the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(cm)
# Confusion Matrix:
# [[317   0   0   0   0]
#  [  0 127   0 186   0]
#  [  0   0 302   7   0]
#  [  0   0   0   0   0]
#  [  9   0   0   0 293]]
