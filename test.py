from ultralytics import YOLO

model = YOLO("yolo11x-pose.pt")

results = model("https://ultralytics.com/images/bus.jpg")
keypoints = results[0].keypoints
print(keypoints.data)