from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from tracking import process_video_with_tracking

app = FastAPI()

VIDEO_DIR = Path("../shared/videos")
OUTPUT_DIR = Path("../shared/csv")
OUTPUT_DIR.mkdir(exist_ok=True)

class VideoRequest(BaseModel):
    filename: str

@app.post("/process")
def process_video(request: VideoRequest):
    input_path = VIDEO_DIR / request.filename
    output_path = OUTPUT_DIR / (request.filename.rsplit('.', 1)[0] + ".csv")

    if not input_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found.")

    try:
        process_video_with_tracking(str(input_path), str(output_path))
        return {"message": "Processing complete", "output_csv": str(output_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
