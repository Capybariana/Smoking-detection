import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

from predict import predict_and_save

app = FastAPI()

VIDEO_DIR = Path("../shared/videos")
CSV_DIR = Path("../shared/csv")
OUTPUT_DIR = Path("../shared/output")
MODEL_PATH = "best_model_100.pth"
OUTPUT_DIR.mkdir(exist_ok=True)


class PredictRequest(BaseModel):
    video_filename: str


@app.post("/predict")
def predict(request: PredictRequest):
    video_path = VIDEO_DIR / request.video_filename
    csv_filename = request.video_filename.rsplit('.', 1)[0] + ".csv"
    csv_path = CSV_DIR / csv_filename
    output_video_path = OUTPUT_DIR / (request.video_filename.rsplit('.', 1)[0] + "_predicted.mp4")

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found.")
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="CSV file not found.")

    try:
        predict_and_save(
            str(video_path),
            str(csv_path),
            MODEL_PATH,
            str(output_video_path),
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        return {"message": "Prediction completed", "output_video": str(output_video_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
