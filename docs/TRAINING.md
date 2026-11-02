# Training Guide

This document covers the full workflow: environment, dataset, leakage-safe preparation, local smoke training, final GPU training on Colab, evaluation, Grad-CAM, and artifact installation.

The repository ships trained **smoke** checkpoints. Final results require running `ml/notebooks/EcoVision_Training_Colab.ipynb` on a GPU. No metric in this project is hand-written.

---

## 1. Environment

Python 3.12 is required; the scientific stack does not have reliable wheels for newer system Pythons on every platform.

```bash
make setup
source .venv/bin/activate     # optional; the Makefile uses .venv/bin/python directly
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

Tested locally with:

- Python 3.12.15
- torch 2.15.0, torchvision 0.29.2 (CPU on macOS arm64)
- NumPy 2.x, Pillow, OpenCV headless, scikit-learn, Matplotlib

Device selection uses `ml.utils.resolve_device`:

```bash
# CPU (default)
python ml/scripts/train_mobilenet.py --device cpu

# CUDA (Colab / Linux GPU)
python ml/scripts/train_mobilenet.py --device cuda

# Apple Silicon GPU (optional; CPU is often comparable for this model size)
python ml/scripts/train_mobilenet.py --device mps
```

All scripts resolve paths relative to the project root, so they work from any working directory.

---

## 2. Dataset download

RealWaste, 4,752 images, ~657 MB.

```bash
# Canonical UCI archive (default)
python ml/scripts/download_dataset.py

# Resumable per-file Hugging Face mirror (recommended on unstable connections)
python ml/scripts/download_dataset.py --source hf --workers 16

# Already-downloaded archive
python ml/scripts/download_dataset.py --zip-path ~/Downloads/realwaste.zip
```

What the script does:

- Streams the download with a size guard and verifies the archive.
- Extracts safely, rejecting absolute paths, `..` traversal, and `__MACOSX` entries.
- Recursively discovers images regardless of zip nesting.
- Normalizes source class names and validates counts against `ml/dataset_meta.json`.
- Fails fast on count mismatches, unmapped classes, or corrupt images; `--allow-count-mismatch` is available only for inspection.
- Writes `ml/datasets/INSPECTION.json` with per-class counts, dimensions, and corrupt-file details.

Expected source counts: Plastic 921, Paper 500, Cardboard 461, Glass 420, Metal 790, Food Organics 411, Vegetation 436, Miscellaneous Trash 495, Textile Trash 318 — total 4,752.

---

## 3. Preparation and leakage-safe splits

```bash
python ml/scripts/prepare_dataset.py
```

Flags: `--seed` (default 42), `--cache-size` (default 256), `--near-duplicate-hamming` (default 5), `--force`.

Steps:

1. Compute SHA-256 for every file; drop exact duplicates.
2. Compute 64-bit dHash; union-find groups images with Hamming distance ≤ 5 so near-duplicate video frames cannot cross splits.
3. Group-aware stratified 70/15/15 split, seed 42.
4. Assert every class appears in every split, with zero SHA overlap and zero dHash-group overlap.
5. Cache 256 × 256 JPEGs in `ml/datasets/processed/` to avoid repeated 524² decoding.

Outputs:

- `ml/datasets/manifest.csv` — path, source class, mapped class, split, hashes
- `ml/datasets/splits.json` — counts, class weights, seed, split hash
- `ml/datasets/processed/` — cached images

Reference split for seed 42:

| Split | Images |
|---|---:|
| Train | 3,327 |
| Validation | 712 |
| Test | 713 |

Split hash: `3ed206b0169d788d6d99be9a8deacde22791951a1c3688e3472ffd3b713708c9`

Class weights: Plastic 0.7369, Paper 1.3580, Cardboard 1.4715, Glass 1.6166, Metal 0.8595, Organic 0.8015, General 0.8353.

Integrity tests:

```bash
pytest -q tests/test_splits.py
```

These tests skip when the dataset is absent; run them after preparation.

---

## 4. Augmentation and class imbalance

Training (`ml/transforms.py`):

- `RandomResizedCrop(224, scale=(0.7, 1.0))`
- Horizontal flip
- Rotation ±15°
- Color jitter (brightness, contrast, saturation, hue)
- ImageNet normalization
- Light random erasing

Evaluation:

- Resize to 256 (shorter side)
- Center crop 224
- ImageNet normalization

Imbalance is handled with class-weighted cross-entropy. Weights are computed from the training split and capped to avoid extreme gradients. **Macro-F1** is the checkpoint-selection metric; accuracy alone would hide minority-class failures.

---

## 5. Local smoke training

Use this to validate the full pipeline quickly and to make the app demoable. It is real training on real images — just on a small stratified subset.

```bash
make smoke
```

Equivalent:

```bash
python ml/scripts/train_mobilenet.py --smoke
python ml/scripts/train_cnn.py --smoke
python ml/scripts/evaluate.py --stage smoke
```

Default smoke settings: up to 60 images per class, 3 epochs, batch 32, CPU, seed 42, output under `ml/models/smoke/` and `ml/evaluation/smoke/`.

The manifest activates the MobileNetV2 smoke checkpoint when both exist. To override:

```bash
ECOVISION_MODEL_PATH=ml/models/smoke/custom_cnn_smoke.pt make serve
```

Smoke metrics validate plumbing, not model quality. The UI shows a development banner while `stage != "final"`.

---

## 6. Final training on Colab

1. Zip the project **without** `.venv`, `ml/datasets`, and trained weights:

   ```bash
   zip -r ecovision-ai.zip . -x ".venv/*" "ml/datasets/*" "ml/models/*.pt" "ml/models/smoke/*" "data/*" ".git/*"
   ```

2. Open `ml/notebooks/EcoVision_Training_Colab.ipynb` in Google Colab.
3. Select **Runtime → Change runtime type → T4 GPU**.
4. Upload the zip (or set `REPO_URL` in the notebook to clone a private/public remote).
5. Run all cells:

   - GPU check
   - Colab dependency install (opencv, scikit-learn, matplotlib, pillow)
   - `download_dataset.py --source hf`
   - `prepare_dataset.py`
   - `train_mobilenet.py --stage-a-epochs 4 --stage-b-epochs 12 --device cuda`
   - `train_cnn.py --epochs 30 --device cuda`
   - `evaluate.py --stage final --device cuda`
   - Grad-CAM samples (2 per class)
   - Artifact packaging and download

Expected T4 runtime: roughly 25–40 minutes for MobileNetV2, 15–25 minutes for the custom CNN, plus download/preparation.

The notebook downloads `ecovision_final_artifacts.zip`:

```
models/mobilenet_v2_best.pt
models/custom_cnn_best.pt
models/model_manifest.json
evaluation/final/metrics.json
evaluation/final/classification_report_*.txt
evaluation/final/confusion_matrix_*.png
evaluation/final/training_curves_*.png
evaluation/final/comparison.md
evaluation/final/comparison.png
evaluation/final/gradcam_samples/*.png
```

---

## 7. Training commands and flags

### MobileNetV2

```bash
python ml/scripts/train_mobilenet.py \
  --stage-a-epochs 4 --stage-b-epochs 12 \
  --batch-size 32 --lr-head 1e-3 --lr-backbone 1e-4 \
  --weight-decay 1e-4 --dropout 0.3 --unfreeze-blocks 3 \
  --device cuda --num-workers 2 --seed 42 --patience 5
```

Stage A freezes the backbone and trains the head. Stage B unfreezes the last three inverted-residual blocks with a lower backbone learning rate, keeps batch-norm statistics frozen, and uses a plateau scheduler. The best validation macro-F1 checkpoint is saved.

### Custom CNN

```bash
python ml/scripts/train_cnn.py \
  --epochs 30 --batch-size 32 --lr 1e-3 --weight-decay 1e-4 \
  --dropout 0.4 --device cuda --num-workers 2 --seed 42 --patience 5
```

AdamW with a cosine schedule, label smoothing 0.1, gradient clipping, and early stopping.

Both scripts write:

- `ml/models/<arch>_best.pt` (or `ml/models/smoke/<arch>_smoke.pt`)
- `ml/evaluation/<stage>/<arch>_history.json`
- `ml/evaluation/<stage>/<arch>_training_config.json`
- Training curves and a manifest update

Checkpoints contain only tensors plus JSON-serializable metadata; they are loaded with `weights_only=True`.

---

## 8. Evaluation

```bash
# Smoke checkpoints → ml/evaluation/smoke/
python ml/scripts/evaluate.py --stage smoke

# Final checkpoints → ml/evaluation/final/
python ml/scripts/evaluate.py --stage final --device cuda

# One architecture only
python ml/scripts/evaluate.py --stage final --arch mobilenet_v2
```

The script evaluates the held-out test split once and writes:

- `metrics.json` — accuracy, macro/weighted precision/recall/F1, per-class metrics, support, confusion matrix data, parameter counts, checkpoint size, CPU latency, training time, environment
- `classification_report_<arch>.txt`
- `confusion_matrix_<arch>.png`
- `training_curves_<arch>.png`
- `comparison.md` and `comparison.png`

The test split is never used for tuning. Re-running evaluation is safe and deterministic for a fixed checkpoint and split.

---

## 9. Single-image inference and Grad-CAM

```bash
# Top-3 predictions as a table
python ml/scripts/predict.py path/to/image.jpg
python ml/scripts/predict.py path/to/folder --json

# Grad-CAM samples for the test split, two per class
python ml/scripts/gradcam.py --samples-per-class 2 --split test \
  --output ml/evaluation/final/gradcam_samples
```

`predict.py` uses the same checkpoint resolution as the backend. `gradcam.py` writes `<class>_<hash>_<predicted>.png` and `<class>_<hash>_overlay.png` panels.

---

## 10. Installing final artifacts

```bash
python ml/scripts/install_artifacts.py ~/Downloads/ecovision_final_artifacts.zip
```

Validation before copying:

- Archive is intact and contains no unsafe paths.
- Both checkpoints load with `weights_only=True`, declare `stage: final`, and match their filenames' architecture.
- Class order exactly matches `ml/classes.py`.
- `metrics.json` is `stage: final`, contains both architectures, accuracy, macro-F1, a comparison table, and a non-empty test split.
- Manifest points at an installed checkpoint.

On success it copies weights into `ml/models/`, evaluation artifacts into `ml/evaluation/final/`, writes the manifest, and prints a restart reminder. Use `--dry-run` to validate without copying, `--force` to replace existing final artifacts.

```bash
python ml/scripts/install_artifacts.py artifacts.zip --dry-run
```

The installer rejects smoke checkpoints renamed as final, so a development run cannot be mistaken for a final result.

---

## 11. Reproducibility

- Seed 42 is applied to Python, NumPy, and torch; dataloader workers are seeded.
- The split hash is stored in every checkpoint and in `metrics.json`; a checkpoint trained on a different split is detectable.
- Versions (Python, torch, platform) are recorded in `metrics.json`.
- Exact T4 timings will vary with Colab hardware and dataset cache state; accuracy should be close but not bit-identical across GPU types.

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Class order mismatch` | Checkpoint from a different class order | Retrain or reinstall a compatible artifact |
| `stage 'smoke', expected 'final'` | Installing smoke weights as final | Run the full Colab training |
| Validation macro-F1 flat at chance | Frozen backbone + too few epochs, or LR too low | Train the head longer, raise `--lr-head` |
| CUDA out of memory | Batch size too large | Lower `--batch-size` to 16 or 8 |
| Dataset counts differ | Partial download or different archive | Re-download with `--force`; inspect `INSPECTION.json` |
| Split hash differs from documentation | Different dataset version or seed | Use the documented dataset and seed; record the new hash |
| Training is extremely slow on Mac | CPU/MPS limitations | Use Colab for final training |
| `torch.load` unpickling error | Old checkpoint format | Re-run training with the current `ml/artifacts.py` |
