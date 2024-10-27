import cv2
from ultralytics import YOLO, solutions

model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture("../pictures/video_originals/INSERT_THE_NAME_OF_YOUR_PUSHUP_VIDEO.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter("../pictures/video_results/workout.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

gym_object = solutions.AIGym(
    line_thickness=2,
    view_img=True,
    pose_type="pushup",
    kpts_to_check=[6, 8, 10],
)

frame_count = 0
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    frame_count += 1
    results = model.track(im0, verbose=False)  # Tracking recommended
    # results = model.predict(im0)  # Prediction also supported
    im0 = gym_object.start_counting(im0, results, frame_count)
    video_writer.write(im0)

cv2.destroyAllWindows()
video_writer.release()