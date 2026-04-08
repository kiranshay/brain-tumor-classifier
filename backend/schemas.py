from typing import Dict, Optional

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    id: str
    predicted_class: str
    confidence: float
    all_confidences: Dict[str, float]
    inference_time_ms: float
    original_filename: str
    subtype: Optional[str] = None
    subtype_confidence: Optional[float] = None
    subtype_confidences: Optional[Dict[str, float]] = None
    gradcam: Optional[str] = None
    subtype_gradcam: Optional[str] = None


class PredictionHistoryItem(BaseModel):
    id: str
    created_at: str
    predicted_class: str
    confidence: float
    all_confidences: Dict[str, float]
    thumbnail_base64: Optional[str]
    original_filename: Optional[str]
    inference_time_ms: float
    subtype: Optional[str] = None
    subtype_confidence: Optional[float] = None
    subtype_confidences: Optional[Dict[str, float]] = None


class StatsResponse(BaseModel):
    total_predictions: int
    class_distribution: Dict[str, int]
    avg_confidence: float
    avg_inference_time_ms: float
