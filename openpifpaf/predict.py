import cv2
import torch
import pandas as pd
import numpy as np
from model import TemporalTransformer
from preprocess import extract_sequences

def predict_and_save(video_path: str, csv_path: str, model_path: str, output_video_path: str, device: str = 'cpu', seq_len: int = 10):
    device = torch.device(device)
    model = TemporalTransformer(seq_len=seq_len).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    sequences = extract_sequences(video_path, csv_path, seq_len)
    df = pd.read_csv(csv_path)

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 25.0, (int(cap.get(3)), int(cap.get(4))))

    with torch.no_grad():
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_idx in range(frame_count):
            success, frame = cap.read()
            if not success:
                break

            people = df[df['frame'] == frame_idx]

            for _, person in people.iterrows():
                pid = person['id']
                x, y, w, h = person['x'], person['y'], person['width'], person['height']

                person_sequence = None
                for seq, _ , seq_frame in sequences:
                    if seq_frame == frame_idx:
                        for person_data in seq:
                            person_id = person_data[0]
                            if person_id == pid:
                                person_sequence = seq
                                break

                if person_sequence is None:
                    continue

                person_sequence = [person_data[2] for person_data in person_sequence]
                person_sequence = np.array(person_sequence)
                person_sequence = torch.tensor(person_sequence, dtype=torch.float32).unsqueeze(0).to(device)

                output = model(person_sequence)
                _, predicted_class = torch.max(output, dim=1)
                predicted_label = predicted_class.item()

                color = (0, 0, 255) if predicted_label == 1 else (0, 255, 0)
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
                text = "Smoking" if predicted_label == 1 else "Not Smoking"
                cv2.putText(frame, text, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            out.write(frame)

    cap.release()
    out.release()
    print(f"Processed video saved to: {output_video_path}")
