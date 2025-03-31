import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt  # Добавлен для визуализации


# Определение устройства (GPU, если доступен, иначе CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class SmokingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        folder_name = os.path.basename(data_dir).lower()
        if folder_name == 'smoke':
            self.label = 1
        elif folder_name == 'not_smoking':
            self.label = 0
        else:
            raise ValueError(f"Unexpected folder name: {folder_name}. Expected 'smoke' or 'not_smoking'.")
        
        print(f"Folder: {folder_name}, Label: {self.label}")
        
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(data_dir, filename)
                with open(file_path, 'r') as f:
                    for line in f:
                        try:
                            frame_data = json.loads(line.strip())
                            for prediction in frame_data['predictions']:
                                # Преобразуем ключевые точки в правильный формат
                                keypoints = prediction['keypoints']
                                if isinstance(keypoints, dict):
                                    # Если ключевые точки представлены как словарь (например, COCO format)
                                    keypoints = self._convert_keypoints_dict_to_list(keypoints)
                                elif not isinstance(keypoints[0], (list, tuple)):
                                    # Если ключевые точки представлены как плоский список
                                    keypoints = self._convert_flat_keypoints(keypoints)
                                
                                self.samples.append({
                                    'keypoints': keypoints,
                                    'bbox': prediction['bbox'],
                                    'score': prediction['score'],
                                    'label': self.label
                                })
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON in file {file_path}: {e}")

    def _convert_keypoints_dict_to_list(self, keypoints_dict):
        """Конвертирует ключевые точки из формата словаря в список"""
        # Пример для COCO format: {'nose': [x,y,score], 'left_eye': [x,y,score], ...}
        return list(keypoints_dict.values())
    
    def _convert_flat_keypoints(self, flat_keypoints):
        """Конвертирует плоский список ключевых точек в список списков"""
        # Пример: [x1,y1,score1, x2,y2,score2, ...] -> [[x1,y1,score1], [x2,y2,score2], ...]
        return [flat_keypoints[i:i+3] for i in range(0, len(flat_keypoints), 3)]

    def __len__(self):
        return len(self.samples)
    
    def normalize_keypoints(self, keypoints, bbox):
        """Нормализует ключевые точки относительно bounding box"""
        x, y, w, h = bbox
        normalized_kps = []
        
        for kp in keypoints:
            # Проверяем формат ключевой точки
            if isinstance(kp, (list, tuple)) and len(kp) >= 2:
                # Нормализуем координаты x и y относительно bbox
                nx = (kp[0] - x) / w if w > 0 else 0.0
                ny = (kp[1] - y) / h if h > 0 else 0.0
                # Оставляем score как есть (он уже нормализован)
                ns = kp[2] if len(kp) > 2 else 1.0
                normalized_kps.extend([nx, ny, ns])
            else:
                # Если формат неправильный, добавляем нули
                normalized_kps.extend([0.0, 0.0, 0.0])
        
        return normalized_kps
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Нормализуем ключевые точки
        normalized_kps = self.normalize_keypoints(sample['keypoints'], sample['bbox'])
        
        if self.transform:
            normalized_kps = self.transform(normalized_kps)
            
        return {
            'keypoints': torch.tensor(normalized_kps, dtype=torch.float32),
            'bbox': torch.tensor(sample['bbox'], dtype=torch.float32),
            'score': torch.tensor(sample['score'], dtype=torch.float32),
            'label': torch.tensor(sample['label'], dtype=torch.long)  # Изменено на long для совместимости с loss
        }


class SmokingClassifier(nn.Module):
    def __init__(self, input_size, hidden_layers=[256, 128], dropout_prob=0.3):
        """
        Args:
            input_size: размер входных данных (количество ключевых точек * 3 для x, y, score)
            hidden_layers: список с размерами скрытых слоев
            dropout_prob: вероятность dropout
        """
        super(SmokingClassifier, self).__init__()
        
        layers = []
        in_features = input_size
        
        # Создаем скрытые слои
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            in_features = hidden_size
        
        # Выходной слой
        layers.append(nn.Linear(in_features, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        # Преобразуем входные данные в плоский вектор
        x = x.view(x.size(0), -1)  # размерность (batch_size, input_size)
        
        # Пропускаем через модель
        x = self.model(x)
        
        # Применяем сигмоиду для получения вероятности
        return torch.sigmoid(x)


def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    """
    Функция для обучения модели
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Обучение
        for batch in train_loader:
            keypoints = batch['keypoints'].to(device)
            labels = batch['label'].float().unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            outputs = model(keypoints)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Средняя потеря за эпоху
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Валидация
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                keypoints = batch['keypoints'].to(device)
                labels = batch['label'].float().unsqueeze(1).to(device)
                
                outputs = model(keypoints)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses


def predict(model, dataloader, device=device, threshold=0.5):
    """
    Функция для предсказания на данных
    """
    model.eval()
    all_predictions = []
    all_confidences = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            keypoints = batch['keypoints'].to(device)
            labels = batch['label'].cpu().numpy()
            
            outputs = model(keypoints)
            probs = outputs.squeeze().cpu().numpy()
            preds = (probs >= threshold).astype(int)
            
            all_predictions.extend(preds)
            all_confidences.extend(probs)
            all_labels.extend(labels)
    
    return all_predictions, all_confidences, all_labels

def evaluate_predictions(true_labels, predictions, confidences):
    """
    Функция для оценки качества предсказаний
    """
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=['not_smoking', 'smoke']))
    
    return accuracy

def plot_losses(train_losses, val_losses):
    """Визуализация кривых обучения"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def collate_fn(batch):
    """
    Функция для объединения данных в батчи
    """
    return {
        'keypoints': torch.stack([item['keypoints'] for item in batch]),
        'bbox': torch.stack([item['bbox'] for item in batch]),
        'score': torch.stack([item['score'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch])
    }


def main():
    # Загрузка данных
    smoke_dataset = SmokingDataset(data_dir='dataset/labels/smoke')
    not_smoking_dataset = SmokingDataset(data_dir='dataset/labels/not_smoking')
    
    # Объединение датасетов
    full_dataset = ConcatDataset([smoke_dataset, not_smoking_dataset])
    
    # Разделение на train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Создание DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Создание модели
    input_size = 17 * 3  # 17 ключевых точек * 3 значения (x, y, score)
    model = SmokingClassifier(input_size).to(device)
    
    # Обучение модели
    print("Starting training...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=15)
    
    # Визуализация потерь
    plot_losses(train_losses, val_losses)
    
    # Загрузка лучшей модели
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))  # Исправлено предупреждение
    
    # Предсказание на валидационных данных
    print("\nEvaluating on validation set...")
    preds, confidences, true_labels = predict(model, val_loader)
    evaluate_predictions(true_labels, preds, confidences)
    
    # Пример предсказания для одного человека
    print("\nExample prediction for a single person:")
    sample = val_dataset[0]
    single_kps = sample['keypoints'].unsqueeze(0).to(device)
    confidence = model(single_kps).item()
    print(f"True label: {sample['label'].item()}, Predicted confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()