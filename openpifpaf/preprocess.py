import cv2
import pandas as pd
import torch
from collections import defaultdict, deque
from typing import List, Tuple
import openpifpaf


class SequenceCollector:
    def __init__(self, seq_len: int = 10):
        self.seq_len = seq_len
        self.buffers = defaultdict(lambda: deque(maxlen=seq_len))
        self.labels = {}  # Сохраняем метки и последний кадр

    def update(self, person_id, keypoints, label, frame):
        # Добавляем кадр и ключевые точки в буфер
        self.buffers[person_id].append((frame, keypoints))
        # Сохраняем метку и кадр
        self.labels[person_id] = (label, frame)

    def ready(self, person_id):
        return len(self.buffers[person_id]) == self.seq_len

    def get_sequence(self, person_id):
        # Формируем последовательность вида [id, frame, [17, 3]]
        sequence = []
        for frame, keypoints in self.buffers[person_id]:
            sequence.append([person_id, frame, keypoints])
        # Добавляем информацию о метке и последнем кадре
        label, last_frame = self.labels[person_id]
        return sequence, label, last_frame  # Возвращаем также метку и последний кадр


def extract_sequences(video_path: str, csv_path: str, seq_len: int = 10) -> List[Tuple[List, int]]:  # меняем тип возвращаемого значения
    df = pd.read_csv(csv_path)

    # Словарь для кадров
    frame_dict = defaultdict(list)
    for _, row in df.iterrows():
        frame = int(row['frame'])
        person_id = int(row['id'])
        bbox = [row['x'], row['y'], row['width'], row['height']]
        label = int(row['label'])
        frame_dict[frame].append({'id': person_id, 'bbox': bbox, 'label': label})

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    predictor = openpifpaf.Predictor(checkpoint='resnet50')  # используйте модель по умолчанию

    collector = SequenceCollector(seq_len=seq_len)
    samples = []

    for frame_idx in range(total_frames):
        success, frame = cap.read()
        if not success:
            break

        people = frame_dict.get(frame_idx, [])
        for person in people:
            pid = person['id']
            x, y, w, h = person['bbox']
            label = person['label']

            # Получаем crop для данного человека
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # Получаем предсказания от openpifpaf
            predictions, _, _ = predictor.numpy_image(crop)
            if not predictions:
                continue

            keypoints = predictions[0].data  # (17, 3)

            # Нормализация ключевых точек
            keypoints[:, 0] /= w
            keypoints[:, 1] /= h

            # Обновляем буфер с данными
            collector.update(pid, keypoints, label, frame_idx)

            # Когда последовательность готова, сохраняем её
            if collector.ready(pid):
                seq, label, last_frame = collector.get_sequence(pid)
                samples.append((seq, label, last_frame))

    cap.release()
    return samples

