from ultralytics import YOLO
import cv2

# モデル読み込み
model = YOLO("best.pt")

# 推論実行（画像ファイルを指定）
results = model("ex3.jpg", conf=0.7)[0]

# クラス名（アノテーション時に設定した順）
class_names = ['BVB', 'BAR', 'ゴールキーパー', '審判']

# 色指定（BGR形式）
color_map = {
    'BAR': (255, 0, 0),     # 赤
    'BVB': (0, 255, 255),   # 黄色
}

# 元画像の取得
img = results.orig_img

# ボックス描画
for box in results.boxes:
    cls_id = int(box.cls.item())
    cls_name = class_names[cls_id]

    if cls_name in color_map:
        x1, y1, x2, y2 = map(int, box.xyxy.squeeze().tolist())
        cv2.rectangle(img, (x1, y1), (x2, y2), color_map[cls_name], thickness=3)

# 結果表示
cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()