import os
import json
import cv2
import torch
import numpy as np
from tqdm import tqdm

from DetectorBase import SmokingClassifier, SmokingDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VideoSmokingDetector:
    def __init__(self, model_path, input_size=51):
        self.model = SmokingClassifier(input_size).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        self.model.eval()
        
    def process_frame(self, frame, predictions):
        """
        Обрабатывает кадр: рисует bbox и классифицирует курящих/не курящих
        """
        for pred in predictions:
            bbox = pred['bbox']  # [x, y, w, h]
            keypoints = pred['keypoints']
            
            # Проверяем и преобразуем формат keypoints
            if isinstance(keypoints, dict):
                # Если ключевые точки представлены как словарь
                keypoints = list(keypoints.values())
            elif not isinstance(keypoints[0], (list, tuple)):
                # Если ключевые точки представлены как плоский список
                keypoints = [keypoints[i:i+3] for i in range(0, len(keypoints), 3)]
            
            # Предсказание
            normalized_kps = self._normalize_keypoints(keypoints, bbox)
            input_tensor = torch.tensor(normalized_kps, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                confidence = self.model(input_tensor).item()
            
            # Цвет bbox
            color = (0, 0, 255) if confidence > 0.5 else (0, 255, 0)
            label = "Smoking" if confidence > 0.5 else "Not Smoking"
            
            # Рисуем bbox
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Добавляем текст
            cv2.putText(frame, f"{label}: {confidence:.2f}", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def _normalize_keypoints(self, keypoints, bbox):
        """Нормализация ключевых точек"""
        x, y, w, h = bbox
        normalized = []
        for kp in keypoints:
            if isinstance(kp, (list, tuple)) and len(kp) >= 2:
                nx = (kp[0] - x) / w if w > 0 else 0.0
                ny = (kp[1] - y) / h if h > 0 else 0.0
                ns = kp[2] if len(kp) > 2 else 1.0
                normalized.extend([nx, ny, ns])
            else:
                normalized.extend([0.0, 0.0, 0.0])
        return normalized
    
    def process_video(self, video_path, json_path, output_path):
        """
        Обрабатывает видео с использованием данных из JSON
        """
        # Загрузка JSON с предсказаниями
        with open(json_path) as f:
            frame_predictions = [json.loads(line) for line in f]
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Находим предсказания для текущего кадра
            current_frame_preds = next(
                (fp for fp in frame_predictions if fp['frame'] == frame_idx), 
                {'predictions': []}
            )
            
            try:
                processed_frame = self.process_frame(frame.copy(), current_frame_preds['predictions'])
                out.write(processed_frame)
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {str(e)}")
                out.write(frame)  # Сохраняем оригинальный кадр в случае ошибки
            
            frame_idx += 1
            pbar.update(1)
        
        cap.release()
        out.release()
        pbar.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--json', type=str, required=True, help='JSON with predictions')
    parser.add_argument('--output', type=str, required=True, help='Output video path')
    parser.add_argument('--model', type=str, default='best_model.pth', help='Path to trained model')
    
    args = parser.parse_args()
    
    detector = VideoSmokingDetector(args.model)
    detector.process_video(args.video, args.json, args.output)