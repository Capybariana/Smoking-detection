import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
import random
import numpy as np
import os

from model import TemporalTransformer, KeypointSequenceDataset

# ======= Настройки =======
video_path = 'data/video.mp4'
csv_path = 'data/annotations.csv'
seq_len = 10
batch_size = 16
learning_rate = 1e-4
num_epochs = 100
save_path = './best_model_100.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======= Фиксируем зерно для воспроизводимости =======
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed()

# ======= Датасеты =======
full_dataset = KeypointSequenceDataset(video_path=video_path, csv_path=csv_path, seq_len=seq_len)

# 80% train, 20% val
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ======= Модель =======
model = TemporalTransformer(
    seq_len=seq_len,
    num_joints=17,
    joint_dim=3,
    num_classes=2,
    d_model=128,
    nhead=4,
    num_layers=2
).to(device)

# ======= Оптимизатор и функция потерь =======
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ======= Шедулер (уменьшает LR, если метрика не улучшается) =======
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

best_acc = 0.0


# ======= Тренировка на одной эпохе =======
def run_epoch(model, dataloader, criterion, optimizer=None, train=True):
    model.train() if train else model.eval()
    epoch_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(train):
        for seq, label in dataloader:
            seq, label = seq.to(device), label.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(seq)
            loss = criterion(outputs, label)

            if train:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    avg_loss = epoch_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


# ======= Основной тренировочный цикл =======
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, save_path):
    global best_acc
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, train=False)

        scheduler.step(val_acc)  # Изменяем learning rate по валидационной метрике

        print(f"Train     Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
        print(f"Val       Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")
        print(f"LR        Now : {optimizer.param_groups[0]['lr']:.6f}")

        # Сохраняем лучшую модель
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ New best model saved with val_acc = {best_acc:.4f}")


# ======= Запуск =======
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, save_path)
print(f"\n🎯 Training complete. Best validation accuracy: {best_acc:.4f}")
