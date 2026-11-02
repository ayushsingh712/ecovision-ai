# API Reference

Base URL: `http://127.0.0.1:8000/api`

All responses are JSON. Requests may include `X-Request-ID`; the server echoes it (and generates one if absent). Errors use the envelope below. Interactive OpenAPI docs are served at `/docs`.

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed.",
    "detail": null,
    "request_id": "2f6c1a9b7d4e0c31"
  }
}
```

| Status | Codes |
|---|---|
| 404 | `PREDICTION_NOT_FOUND`, `NOT_FOUND` |
| 413 | `FILE_TOO_LARGE` |
| 415 | `UNSUPPORTED_IMAGE_TYPE` |
| 422 | `EMPTY_FILE`, `INVALID_IMAGE`, `ANIMATED_IMAGE`, `IMAGE_TOO_SMALL`, `IMAGE_TOO_LARGE_PIXELS`, `VALIDATION_ERROR` |
| 503 | `MODEL_UNAVAILABLE` |
| 504 | `INFERENCE_TIMEOUT` |
| 500 | `INTERNAL_ERROR` |

---

## GET /api/health

Service status. Returns `200` even when no checkpoint is installed (`status: "degraded"`).

```bash
curl http://127.0.0.1:8000/api/health
```

```json
{
  "status": "ok",
  "app": "EcoVision AI",
  "version": "1.0.0",
  "environment": "development",
  "model_loaded": true,
  "model_stage": "smoke",
  "model_version": "realwaste-7class-v1",
  "device": "cpu",
  "database": "ok",
  "uptime_seconds": 182.41,
  "timestamp": "2026-11-01T16:52:03+00:00"
}
```

---

## POST /api/predict

Multipart upload with a single `image` field. Accepted: JPEG, PNG, WebP; max 8 MB; at least 64 × 64; one frame; at most 40 megapixels.

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -F "image=@sample.jpg;type=image/jpeg"
```

`200 OK`

```json
{
  "prediction_id": "9c1f4a7e2b8d4c6e9a0f3b5d7c1e2a4b",
  "filename": "sample.jpg",
  "predicted_class": "Plastic",
  "confidence": 0.6755,
  "top3": [
    { "label": "Plastic", "confidence": 0.6755 },
    { "label": "Metal", "confidence": 0.1812 },
    { "label": "Paper", "confidence": 0.0714 }
  ],
  "advice": {
    "bin_type": "Mixed recycling (rigid plastics), where accepted",
    "bin_color": "Often blue, but varies by municipality",
    "disposal_method": "Empty, rinse, and dry rigid plastic containers…",
    "instructions": ["Empty and rinse food or drink residue.", "…"],
    "environmental_impact": "Recycling plastic displaces some virgin fossil-based resin…",
    "examples": ["Beverage bottles", "Food tubs", "…"],
    "cautions": ["Compostable or biodegradable plastics are usually not accepted…"],
    "scope": "general",
    "region_note": "Bin colors and accepted items vary widely by municipality and country; verify locally.",
    "disclaimer": "This guide is general education, not a legal or municipal ruling…",
    "sources": [{ "title": "U.S. EPA — Recycling Basics and Resources", "url": "https://www.epa.gov/recycle" }]
  },
  "gradcam": {
    "original_url": "/media/9c1f4a7e2b8d4c6e9a0f3b5d7c1e2a4b/original.jpg",
    "heatmap_url": "/media/9c1f4a7e2b8d4c6e9a0f3b5d7c1e2a4b/heatmap.png",
    "overlay_url": "/media/9c1f4a7e2b8d4c6e9a0f3b5d7c1e2a4b/overlay.png"
  },
  "model_stage": "smoke",
  "model_version": "realwaste-7class-v1",
  "created_at": "2026-11-01T16:52:11+00:00",
  "low_confidence": false
}
```

Notes:

- `top3` is sorted by confidence and always has up to three entries.
- `low_confidence` is `true` when the top probability is below `0.60`.
- `/api/predict` performs one forward pass and returns as soon as classification, persistence, and the original thumbnail are complete. Grad-CAM is not generated during this request.
- `original_url` is valid immediately. `heatmap_url` and `overlay_url` are populated by the explanation endpoint below; requesting them before generation returns `404`.
- No original upload is retained; only the normalized 512 × 512 display image is stored, plus Grad-CAM outputs after an explanation succeeds.
- Prediction is unavailable (`503`) if no checkpoint resolves at startup.

---

## POST /api/predictions/{prediction_id}/explanation

Generates Grad-CAM outputs for an existing prediction on demand. The classification response has already been returned, so a slow or failed explanation never invalidates it.

```bash
curl -X POST http://127.0.0.1:8000/api/predictions/9c1f4a7e2b8d4c6e9a0f3b5d7c1e2a4b/explanation
```

`200 OK` — `status` is `ready`, `disabled`, or `failed`.

```json
{
  "prediction_id": "9c1f4a7e2b8d4c6e9a0f3b5d7c1e2a4b",
  "status": "ready",
  "gradcam": {
    "original_url": "/media/9c1f4a7e2b8d4c6e9a0f3b5d7c1e2a4b/original.jpg",
    "heatmap_url": "/media/9c1f4a7e2b8d4c6e9a0f3b5d7c1e2a4b/heatmap.png",
    "overlay_url": "/media/9c1f4a7e2b8d4c6e9a0f3b5d7c1e2a4b/overlay.png"
  },
  "message": null
}
```

| `status` | Meaning |
|---|---|
| `ready` | `heatmap.png` and `overlay.png` exist and are served from `/media/...`. Repeat calls reuse the files instead of regenerating them. |
| `disabled` | `ECOVISION_GRADCAM_ENABLED=false`. `gradcam` is `null`. |
| `failed` | Generation exceeded `ECOVISION_GRADCAM_TIMEOUT_SECONDS` or raised. `gradcam` is `null`; the prediction and its stored original remain valid. |

The explanation uses the stored 512 × 512 image and the class that was originally predicted; it does not re-run classification. `404 PREDICTION_NOT_FOUND` is returned for an unknown id and `503 MODEL_UNAVAILABLE` when no checkpoint is loaded.

---

## POST /api/feedback

One feedback row per prediction. Posting again for the same `prediction_id` updates it (upsert).

```bash
curl -X POST http://127.0.0.1:8000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_id": "9c1f4a7e2b8d4c6e9a0f3b5d7c1e2a4b",
    "is_correct": false,
    "corrected_class": "Metal",
    "comment": "This is an aluminium can."
  }'
```

| Field | Type | Required | Rules |
|---|---|---|---|
| `prediction_id` | string | yes | 8–64 chars, must exist |
| `is_correct` | boolean | yes | — |
| `corrected_class` | string \| null | no | One of the seven classes; must be omitted when `is_correct` is true |
| `comment` | string \| null | no | Trimmed, max 1000 chars |

```json
{
  "prediction_id": "9c1f4a7e2b8d4c6e9a0f3b5d7c1e2a4b",
  "is_correct": false,
  "corrected_class": "Metal",
  "comment": "This is an aluminium can.",
  "verified": false,
  "created_at": "2026-11-01T16:53:02+00:00"
}
```

`verified` is always `false` for user-submitted feedback; manual verification is not exposed in this version.

---

## GET /api/history

Most recent predictions, newest first.

| Query | Default | Bounds |
|---|---|---|
| `limit` | 20 | 1–100 |
| `offset` | 0 | ≥ 0 |

```bash
curl "http://127.0.0.1:8000/api/history?limit=10&offset=0"
```

```json
{
  "items": [
    {
      "prediction_id": "9c1f4a7e2b8d4c6e9a0f3b5d7c1e2a4b",
      "filename": "sample.jpg",
      "predicted_class": "Plastic",
      "confidence": 0.6755,
      "top3": [{ "label": "Plastic", "confidence": 0.6755 }],
      "image_url": "/media/9c1f4a7e2b8d4c6e9a0f3b5d7c1e2a4b/original.jpg",
      "created_at": "2026-11-01T16:52:11+00:00",
      "model_stage": "smoke",
      "feedback": {
        "is_correct": false,
        "corrected_class": "Metal",
        "comment": "This is an aluminium can.",
        "verified": false,
        "created_at": "2026-11-01T16:53:02+00:00"
      }
    }
  ],
  "total": 42,
  "limit": 10,
  "offset": 0
}
```

`feedback` is `null` when no feedback exists.

---

## GET /api/statistics

Aggregates for the dashboard. All seven classes are returned, including zero counts.

```bash
curl http://127.0.0.1:8000/api/statistics
```

```json
{
  "total_predictions": 42,
  "average_confidence": 0.6812,
  "most_common_class": "Metal",
  "class_distribution": [
    { "label": "Plastic", "count": 7, "percentage": 16.67 },
    { "label": "Paper", "count": 5, "percentage": 11.9 },
    { "label": "Cardboard", "count": 4, "percentage": 9.52 },
    { "label": "Glass", "count": 6, "percentage": 14.29 },
    { "label": "Metal", "count": 9, "percentage": 21.43 },
    { "label": "Organic", "count": 6, "percentage": 14.29 },
    { "label": "General", "count": 5, "percentage": 11.9 }
  ],
  "feedback": {
    "total": 11,
    "correct": 8,
    "incorrect": 3,
    "user_reported_accuracy": 0.7273
  },
  "daily": [
    { "date": "2026-10-26", "count": 0 },
    { "date": "2026-10-27", "count": 2 },
    { "date": "2026-11-01", "count": 40 }
  ],
  "model_stage": "smoke"
}
```

`average_confidence`, `most_common_class`, and `user_reported_accuracy` are `null` when there is no data. `daily` always covers the last seven UTC days.

---

## GET /api/model-info

Metadata for the About page: active checkpoint, both architecture descriptions, class order, dataset mapping/attribution, evaluation metrics, comparison table, and limitations.

```bash
curl http://127.0.0.1:8000/api/model-info
```

```json
{
  "model_loaded": true,
  "model_name": "MobileNetV2 (transfer learning)",
  "model_stage": "smoke",
  "model_version": "realwaste-7class-v1",
  "is_final": false,
  "device": "cpu",
  "checkpoint": "mobilenet_v2_smoke.pt",
  "trained_at": "2026-11-01T16:49:07+00:00",
  "architecture": {
    "name": "mobilenet_v2",
    "display_name": "MobileNetV2 (transfer learning)",
    "parameters": 2232839,
    "input_size": 224,
    "target_layer": "features[-1]",
    "description": "MobileNetV2 initialized with ImageNet weights…"
  },
  "architectures": [
    {
      "name": "custom_cnn",
      "display_name": "Custom CNN (from scratch)",
      "parameters": 1175015,
      "input_size": 224,
      "target_layer": "features[-1] (final ConvBlock)",
      "description": "A four-block convolutional network…"
    },
    {
      "name": "mobilenet_v2",
      "display_name": "MobileNetV2 (transfer learning)",
      "parameters": 2232839,
      "input_size": 224,
      "target_layer": "features[-1]",
      "description": "MobileNetV2 initialized with ImageNet weights…"
    }
  ],
  "classes": [{ "index": 0, "name": "Plastic" }],
  "dataset": {
    "dataset": "RealWaste",
    "license": "CC BY 4.0",
    "doi": "10.24432/C5SS4G",
    "total_images": 4752,
    "target_classes": ["Plastic", "Paper", "Cardboard", "Glass", "Metal", "Organic", "General"],
    "class_mapping": { "Organic": ["Food Organics", "Vegetation"] }
  },
  "metrics": { "stage": "smoke", "test_split": { "images": 420 } },
  "comparison": [
    {
      "architecture": "mobilenet_v2",
      "accuracy": 0.7,
      "macro_f1": 0.694642,
      "parameters": 2232839,
      "cpu_latency_ms": 23.84
    }
  ],
  "limitations": ["The active checkpoint is a development smoke model…"],
  "advice_version": "1.0.0",
  "advice_last_reviewed": "2026-11-01"
}
```

`metrics` is read from `ml/evaluation/<final|smoke>/metrics.json` and mirrors the active checkpoint's stage. When no model is installed, model fields are `null`, `is_final` is `false`, and `limitations` explains the degraded state.

---

## Static media

`GET /media/<prediction-id>/<file>` serves `original.jpg` immediately after classification and `heatmap.png`/`overlay.png` after the explanation endpoint succeeds. These files are derived from the normalized display image, not the original upload.

## CORS

Origins come from `ECOVISION_ALLOWED_ORIGINS` (comma-separated; `*` by default). Allowed methods: `GET`, `POST`, `OPTIONS`. Allowed request headers: `Content-Type`, `X-Request-ID`. Exposed response header: `X-Request-ID`.
