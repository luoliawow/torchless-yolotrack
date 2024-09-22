from collections import defaultdict
import cv2
import numpy as np

from utils import inference, load_track_cfg
from track import BOTSORT, TrackModel


model_path = 'yolov8n.onnx' # model_size: (640, 640)
model = cv2.dnn.readNetFromONNX(model_path)

video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

tracker = TrackModel(model_path)
# 存储追踪历史
track_history = defaultdict(lambda: [])

# 从视频读取一帧
success, frame = cap.read()
w, h, _ = frame.shape
scale = max(w, h) / 640

# 循环遍历视频帧
while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = tracker.track(frame)
        boxes = results.xyxy
        track_ids = results.id.astype(np.int32).tolist()

        for item in zip(boxes, track_ids):
            box, track_id = item
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{track_id}', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        # 展示带注释的帧
        cv2.imshow("torchless Tracking", frame)

        # 如果按下'q'则退出循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 如果视频结束则退出循环
        break
