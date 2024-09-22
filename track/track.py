from utils import inference, load_track_cfg, update_results
from .botsort import BOTSORT
import cv2

track_cfg = load_track_cfg("cfg/botsort.yaml")
tracker = BOTSORT(args=track_cfg, frame_rate=30)

class TrackModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = cv2.dnn.readNetFromONNX(model_path)

    def track(self, frame):
        output = inference(self.model, frame)
        tracks = tracker.update(output, frame)
        if len(tracks) == 0:
            return
        return update_results(tracks[:, :-1], frame.shape[:2])