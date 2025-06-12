import cv2
import torch
from ultralytics import YOLO
import numpy

model = YOLO("yolov8x.pt")
results = model.predict("ex3.jpg", conf=0.1)
img = results[0].orig_img
boxes = results[0].boxes

for box in boxes:
    xy1 = box.data[0][0:2]
    xy2 = box.data[0][2:4]
    x1, y1 = xy1.to(torch.int).tolist()
    x2, y2 = xy2.to(torch.int).tolist()

    top = int(y1 + (y2 - y1) * 0.2)
    bottom = int(y1 + (y2 - y1) * 0.5)
    left = x1
    right = x2
    torso = img[top:bottom, left:right]

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

    low_yellow = numpy.array([20, 100, 100])
    high_yellow = numpy.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv, low_yellow, high_yellow)
    yellow = numpy.sum(mask_yellow > 0) / mask_yellow.size

    low_blue = numpy.array([90, 0, 90])
    high_blue = numpy.array([210, 180, 160])
    mask_blue = cv2.inRange(hsv, low_blue, high_blue)
    blue = numpy.sum(mask_blue > 0) / mask_blue.size

    low_red1 = numpy.array([0, 100, 100])
    high_red1 = numpy.array([10, 255, 255])
    mask_red1 = cv2.inRange(hsv, low_red1, high_red1)

    low_red2 = numpy.array([160, 100, 100])
    high_red2 = numpy.array([179, 255, 255])
    mask_red2 = cv2.inRange(hsv, low_red2, high_red2)

    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    red = numpy.sum(mask_red > 0) / mask_red.size

    foot_y1 = y2
    foot_y2 = min(y2 + 10, img.shape[0])
    foot_region = img[foot_y1:foot_y2, x1:x2]
    hsv_foot = cv2.cvtColor(foot_region, cv2.COLOR_BGR2HSV)

    low_green = numpy.array([35, 50, 50])
    high_green = numpy.array([85, 255, 255])
    mask_green = cv2.inRange(hsv_foot, low_green, high_green)
    green_ratio = numpy.sum(mask_green > 0) / mask_green.size

    if green_ratio > 0.05:
        if yellow > 0.12 and yellow > (blue + red):
            cv2.rectangle(
                img, (x1, y1), 
                (x2, y2), (0, 0, 255), 
                thickness=3
            )  

        elif blue > 0.08 or red > 0.08:
            cv2.rectangle(
                img, (x1, y1), 
                (x2, y2), (255, 0, 0), 
                thickness=3
            ) 

cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()