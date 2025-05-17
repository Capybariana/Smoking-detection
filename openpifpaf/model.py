import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

import math

from preprocess import extract_sequences


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoder, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2)

        pe[:, 0::2] = torch.sin(position * div_term)  # чётные
        pe[:, 1::2] = torch.cos(position * div_term)  # нечётные

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor of shape (B, T, D)
        """
        return x + self.pe[:, :x.size(1)]


class TemporalTransformer(nn.Module):
    def __init__(self, seq_len=10, num_joints=17, joint_dim=3, num_classes=2,
                 d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super(TemporalTransformer, self).__init__()

        self.input_dim = num_joints * joint_dim  # 17 * 3 = 51

        # Входная проекция
        self.input_proj = nn.Linear(self.input_dim, d_model)

        # Позиционное кодирование
        self.pos_encoder = PositionalEncoder(d_model=d_model, max_len=seq_len)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        x: (B, T=10, 17, 3)
        """
        B, T, J, C = x.shape
        x = x.view(B, T, -1)  # (B, T, 51)
        x = self.input_proj(x)  # (B, T, d_model)
        x = self.pos_encoder(x)  # (B, T, d_model)
        x = self.transformer_encoder(x)  # (B, T, d_model)
        x = x.mean(dim=1)  # Mean pooling over time
        return self.classifier(x)  # (B, num_classes)


class KeypointSequenceDataset(Dataset):
    def __init__(self, video_path: str, csv_path: str, seq_len: int = 10):
        # Извлекаем последовательности с помощью функции из preprocess
        self.samples = extract_sequences(video_path, csv_path, seq_len)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        seq, label, _ = self.samples[idx]  # frame нам не нужен

        person_sequence = [person_data[2] for person_data in seq]  # Извлекаем ключевые точки (оставляем только координаты)
        person_sequence = np.array(person_sequence)
        
        seq_tensor = torch.tensor(person_sequence, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return seq_tensor, label_tensor