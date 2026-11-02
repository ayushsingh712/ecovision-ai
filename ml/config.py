"""Project-wide paths and training constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = PROJECT_ROOT / "ml"
DATASETS_DIR = ML_DIR / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
PROCESSED_DIR = DATASETS_DIR / "processed"
DATASET_META_PATH = ML_DIR / "dataset_meta.json"
MANIFEST_CSV = DATASETS_DIR / "manifest.csv"
SPLITS_JSON = DATASETS_DIR / "splits.json"
INSPECTION_JSON = DATASETS_DIR / "INSPECTION.json"

ARCHITECTURES_DIR = ML_DIR / "architectures"
MODELS_DIR = ML_DIR / "models"
SMOKE_MODELS_DIR = MODELS_DIR / "smoke"
MODEL_MANIFEST_PATH = MODELS_DIR / "model_manifest.json"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

EVAL_DIR = ML_DIR / "evaluation"
FINAL_EVAL_DIR = EVAL_DIR / "final"
SMOKE_EVAL_DIR = EVAL_DIR / "smoke"

BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "ecovision.db"

INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SEED = 42
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
NEAR_DUPLICATE_HAMMING = 5
CACHE_SIZE = 256
MODEL_VERSION = "realwaste-7class-v1"


def dataset_meta() -> dict:
    import json

    return json.loads(DATASET_META_PATH.read_text(encoding="utf-8"))
