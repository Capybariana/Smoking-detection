import requests
import time
from pathlib import Path

def run_full_pipeline(video_filename: str) -> str:
    video_name = Path(video_filename).name
    base_name = video_name.rsplit(".", 1)[0]
    csv_filename = base_name + ".csv"

    # Шаг 1: Отправка видео на BoxMot-сервер
    print("➡️ Sending video to BoxMot server...")
    boxmot_response = requests.post(
        "http://localhost:8000/process",
        json={"filename": video_name}
    )
    if boxmot_response.status_code != 200:
        raise RuntimeError(f"BoxMot error: {boxmot_response.json()}")

    print("✅ BoxMot processing completed.")

    # Можно вставить задержку, если нужно (например, если BoxMot выполняется в фоне)
    # time.sleep(2)

    # Шаг 2: Отправка запроса на OpenPifPaf + Transformer сервер
    print("➡️ Sending data to OpenPifPaf+Transformer server...")
    predict_response = requests.post(
        "http://localhost:8001/predict",
        json={"video_filename": video_name}
    )
    if predict_response.status_code != 200:
        raise RuntimeError(f"Prediction error: {predict_response.json()}")

    output_path = predict_response.json()["output_video"]
    print(f"✅ Prediction completed. Output video: {output_path}")

    return output_path
