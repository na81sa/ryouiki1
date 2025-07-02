from ultralytics import YOLO
import cv2

# モデル読み込み
model = YOLO(r"C:\Users\nagis\ryouiki\ex8\runs\detect\train3\weights\best.pt")

# 推論実行
results = model("ex4.jpg", save=True, conf=0.3)[0]  # conf=0.25で絞りすぎない

# クラスIDに対応する名前（0: black, 1: white）
class_names = ['black', 'white']
counts = {'black': 0, 'white': 0}

# カウント処理
for cls_id in results.boxes.cls.cpu().numpy():
    cls_name = class_names[int(cls_id)]
    counts[cls_name] += 1

# 結果表示
print("黒石:", counts['black'])
print("白石:", counts['white'])