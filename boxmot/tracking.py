import cv2
import csv
from pathlib import Path
from ultralytics import YOLO
from boxmot import BotSort
import torch
import numpy as np


def process_video_with_tracking(video_path: str, output_csv_path: str):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Модель YOLOv8
    model = YOLO("yolov8m.pt")
    model.fuse()
    model.to(device)

    # Трекер
    tracker = BotSort(
        reid_weights=Path('osnet_x0_25_msmt17.pt'),
        device=device,
        half=False,
    )

    # Видео
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0

    # CSV файл
    csv_file = open(output_csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'id', 'x', 'y', 'width', 'height', 'label'])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8 инференс
        results = model(frame, verbose=False)[0]
        detections = results.boxes

        if detections is not None and detections.xyxy.shape[0] > 0:
            dets = torch.cat(
                (
                    detections.xyxy.cpu(),
                    detections.conf.view(-1, 1).cpu(),
                    detections.cls.view(-1, 1).cpu()
                ), dim=1
            ).numpy()

            people_detections = [det for det in dets if int(det[5]) == 0]

            if people_detections:
                people_detections = np.array(people_detections)

                tracks = tracker.update(people_detections, frame)

                for track in tracks:
                    x1, y1, x2, y2, track_id, conf, cls_id, _ = track
                    width = x2 - x1
                    height = y2 - y1
                    label = 0
                    csv_writer.writerow([
                        frame_idx,
                        int(track_id),
                        int(x1),
                        int(y1),
                        int(width),
                        int(height),
                        int(label)
                    ])

        frame_idx += 1

    cap.release()
    csv_file.close()
