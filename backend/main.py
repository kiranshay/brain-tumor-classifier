import base64
import io
import os

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from model import load_model, predict, predict_with_gradcam
from database import insert_prediction, get_predictions, get_stats
from schemas import PredictionResponse, PredictionHistoryItem, StatsResponse

app = FastAPI(title="Brain Tumor Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:9090",
        "http://127.0.0.1:9090",
        "https://kiranshay.github.io",
        "https://brain-tumor-classifier-q8kn.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
weights_path = os.environ.get("MODEL_PATH", "tumor_classifier.pth")
load_model(weights_path)


def make_thumbnail_base64(image_bytes: bytes, size: tuple = (128, 128)) -> str:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail(size)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=70)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
async def predict_tumor(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()

    try:
        result = predict(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    thumbnail = make_thumbnail_base64(image_bytes)

    db_data = {
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "all_confidences": result["all_confidences"],
        "thumbnail_base64": thumbnail,
        "original_filename": file.filename,
        "inference_time_ms": result["inference_time_ms"],
    }

    if "subtype" in result:
        db_data["subtype"] = result["subtype"]
        db_data["subtype_confidence"] = result["subtype_confidence"]
        db_data["subtype_confidences"] = result["subtype_confidences"]

    db_row = insert_prediction(db_data)

    return PredictionResponse(
        id=db_row["id"],
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        all_confidences=result["all_confidences"],
        inference_time_ms=result["inference_time_ms"],
        original_filename=file.filename or "unknown",
        subtype=result.get("subtype"),
        subtype_confidence=result.get("subtype_confidence"),
        subtype_confidences=result.get("subtype_confidences"),
    )


@app.post("/gradcam")
async def get_gradcam(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()

    try:
        result = predict_with_gradcam(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM failed: {str(e)}")

    return result


@app.get("/predictions", response_model=list[PredictionHistoryItem])
def list_predictions(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    return get_predictions(limit=limit, offset=offset)


@app.get("/stats", response_model=StatsResponse)
def prediction_stats():
    return get_stats()
