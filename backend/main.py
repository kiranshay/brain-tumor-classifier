import base64
import io

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from neuroscan.config import load_config
from neuroscan.inference import CascadeInference
from neuroscan.logging_setup import setup_logging, get_logger

from database import insert_prediction, get_predictions, get_stats
from schemas import PredictionResponse, PredictionHistoryItem, StatsResponse

# ---- Config + logging --------------------------------------------------------
config = load_config()
setup_logging(config)
log = get_logger("main")

THUMBNAIL_SIZE = tuple(config["thumbnail"]["size"])
THUMBNAIL_QUALITY = config["thumbnail"]["jpeg_quality"]

# ---- FastAPI app -------------------------------------------------------------
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

# ---- Cascade inference engine (loads models at startup) ----------------------
cascade = CascadeInference(config)
log.info("CascadeInference initialized.")


def make_thumbnail_base64(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail(THUMBNAIL_SIZE)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=THUMBNAIL_QUALITY)
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
        result = cascade.predict(image_bytes)
    except Exception as e:
        log.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    thumbnail = make_thumbnail_base64(image_bytes)

    db_data = {
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "all_confidences": result["all_confidences"],
        "thumbnail_base64": thumbnail,
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


@app.post("/uncertainty")
async def predict_uncertainty(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()

    try:
        n_samples = config.get("uncertainty", {}).get("n_samples", 50)
        result = cascade.predict_with_uncertainty(image_bytes, n_samples=n_samples)
    except Exception as e:
        log.exception("Uncertainty inference failed")
        raise HTTPException(status_code=500, detail=f"Uncertainty inference failed: {str(e)}")

    return {
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "all_confidences": result["all_confidences"],
        "uncertainty": result["uncertainty"],
        "variance_per_class": result["variance_per_class"],
        "inference_time_ms": result["inference_time_ms"],
        "n_samples": result["n_samples"],
        "original_filename": file.filename or "unknown",
    }


@app.post("/gradcam")
async def get_gradcam(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()

    try:
        return cascade.predict_with_gradcam(image_bytes)
    except Exception as e:
        log.exception("Grad-CAM failed")
        raise HTTPException(status_code=500, detail=f"Grad-CAM failed: {str(e)}")


@app.get("/predictions", response_model=list[PredictionHistoryItem])
def list_predictions(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    return get_predictions(limit=limit, offset=offset)


@app.get("/stats", response_model=StatsResponse)
def prediction_stats():
    return get_stats()
