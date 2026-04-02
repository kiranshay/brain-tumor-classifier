import os
from supabase import create_client, Client

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_KEY"]
        _client = create_client(url, key)
    return _client


def insert_prediction(data: dict) -> dict:
    client = get_client()
    result = client.table("predictions").insert(data).execute()
    return result.data[0]


def get_predictions(limit: int = 20, offset: int = 0) -> list[dict]:
    client = get_client()
    result = (
        client.table("predictions")
        .select("*")
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )
    return result.data


def get_stats() -> dict:
    client = get_client()
    rows = client.table("predictions").select("predicted_class, confidence, inference_time_ms").execute().data

    if not rows:
        return {
            "total_predictions": 0,
            "class_distribution": {},
            "avg_confidence": 0,
            "avg_inference_time_ms": 0,
        }

    total = len(rows)
    class_distribution = {}
    total_confidence = 0
    total_inference_time = 0

    for row in rows:
        cls = row["predicted_class"]
        class_distribution[cls] = class_distribution.get(cls, 0) + 1
        total_confidence += row["confidence"]
        total_inference_time += row["inference_time_ms"]

    return {
        "total_predictions": total,
        "class_distribution": class_distribution,
        "avg_confidence": round(total_confidence / total, 4),
        "avg_inference_time_ms": round(total_inference_time / total, 2),
    }
