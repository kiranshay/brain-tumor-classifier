from pydantic import BaseModel


class PredictionResponse(BaseModel):
    id: str
    predicted_class: str
    confidence: float
    all_confidences: dict[str, float]
    inference_time_ms: float
    original_filename: str
    subtype: str | None = None
    subtype_confidence: float | None = None
    subtype_confidences: dict[str, float] | None = None
    gradcam: str | None = None
    subtype_gradcam: str | None = None


class PredictionHistoryItem(BaseModel):
    id: str
    created_at: str
    predicted_class: str
    confidence: float
    all_confidences: dict[str, float]
    thumbnail_base64: str | None
    original_filename: str | None
    inference_time_ms: float
    subtype: str | None = None
    subtype_confidence: float | None = None
    subtype_confidences: dict[str, float] | None = None


class StatsResponse(BaseModel):
    total_predictions: int
    class_distribution: dict[str, int]
    avg_confidence: float
    avg_inference_time_ms: float
