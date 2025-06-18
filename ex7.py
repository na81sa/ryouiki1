from ultralytics import YOLO
import cv2

model = YOLO("othello.pt")

img_path = "ex4.jpg"
results = model(img_path)[0] 


white_count = 0
black_count = 0

white_min_area = 4600
white_max_area = 9500
white_min_conf = 0.25

black_min_area = 5000
black_max_area = 11000
black_min_conf = 0.35


for box in results.boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

    w = x2 - x1
    h = y2 - y1
    area = w * h

    if cls == 0:
        if area < white_min_area or area > white_max_area or conf < white_min_conf:
            continue
        white_count += 1

    elif cls == 1:
        if area < black_min_area or area > black_max_area or conf < black_min_conf:
            continue
        black_count += 1
        
print(f"白石の数: {white_count}")
print(f"黒石の数: {black_count}")
