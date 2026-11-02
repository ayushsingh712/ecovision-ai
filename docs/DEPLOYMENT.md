# Deployment Guide

EcoVision AI runs as a single FastAPI process that serves the API and, optionally, the static frontend. It needs a trained checkpoint, a writable directory for SQLite and derived media, and CPU (GPU is optional).

## Deployment options

| Option | Frontend | Backend | Persistence |
|---|---|---|---|
| Local development | FastAPI static mount | `uvicorn` | `data/` on disk |
| Single container | FastAPI static mount | Docker | Volumes for `data/` and `ml/models` |
| Split hosting | Static host / CDN | Container or PaaS | Persistent disk for `data/` |
| Managed PaaS | Either | Render / Railway / Fly.io | Attached persistent volume |

## Model artifacts

The Docker image intentionally does **not** bake in model weights. Provide them at runtime:

- `ml/models/model_manifest.json` — points at the active checkpoint
- `ml/models/<checkpoint>.pt` — the weights
- `ml/evaluation/<stage>/metrics.json` — optional, shown on the About page

Train locally (`make smoke`) or run the Colab notebook for final artifacts, then install:

```bash
python ml/scripts/install_artifacts.py ~/Downloads/ecovision_final_artifacts.zip
```

The installer validates stage, architecture, class order, manifest consistency, and metric completeness before copying. After installation, restart the process so `ModelService` picks up the new manifest.

Resolution order at startup:

1. `ECOVISION_MODEL_PATH` if set
2. `model_manifest.json` → `active`
3. `mobilenet_v2_best.pt` → `custom_cnn_best.pt`
4. `smoke/mobilenet_v2_smoke.pt` → `smoke/custom_cnn_smoke.pt`

If none resolves, the API starts in **degraded mode**: `/api/health` returns `"status": "degraded"` and `/api/predict` returns `503 MODEL_UNAVAILABLE`. No fake predictions are produced.

## Local

```bash
cp .env.example .env
make setup
make download
make prepare
make smoke
make serve          # http://127.0.0.1:8000
```

Production-style process without `--reload`:

```bash
.venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Use **one worker**. Each worker loads its own copy of the model and opens its own SQLite connection; multiple workers multiply memory use and can contend on writes. Scale vertically first, then move media/state to shared services (see “Scaling” below).

## Docker

```bash
docker build -t ecovision-ai .
docker compose up -d
docker compose logs -f
curl http://127.0.0.1:8000/api/health
```

Compose mounts:

- `./ml/models:/app/ml/models:ro`
- `./ml/evaluation:/app/ml/evaluation:ro`
- `./data:/app/data`

Without Compose:

```bash
docker run -d --name ecovision \
  -p 8000:10000 \
  -v "$PWD/ml/models:/app/ml/models:ro" \
  -v "$PWD/ml/evaluation:/app/ml/evaluation:ro" \
  -v "$PWD/data:/app/data" \
  -e PORT=10000 \
  -e ECOVISION_ALLOWED_ORIGINS="https://your-frontend.example.com" \
  ecovision-ai:latest
```

The image listens on `0.0.0.0` and reads `PORT` (default `10000`); map the host port you prefer to the container port. Compose maps `8000:10000`.

Image notes:

- `python:3.12-slim`, CPU-only PyTorch wheels, non-root `app` user
- `HEALTHCHECK` calls `/api/health` every 30 s using `PORT`
- No PyPI, dataset, or Hugging Face credentials are needed at build time; a Hub token is only required at runtime for a private model repository
- Build context excludes `.venv`, datasets, weights, tests, and `data/` (see `.dockerignore`)

## Split frontend and backend

The frontend is static HTML/CSS/JS; it can be served from S3/CloudFront, Netlify, Nginx, or a CDN while the API runs elsewhere.

1. Point the frontend at the API by defining `window.ECOVISION_CONFIG` **before** `js/config.js` loads. Add this to each page immediately before the main module script:

   ```html
   <script>
     window.ECOVISION_CONFIG = { apiBase: "https://api.example.com" };
   </script>
   <script type="module" src="js/nav.js"></script>
   ```

   `apiBase` is the origin only; the frontend appends `/api`.

2. Set the backend origin allow-list:

   ```bash
   ECOVISION_ALLOWED_ORIGINS=https://app.example.com
   ```

3. Set `ECOVISION_SERVE_FRONTEND=false` on the API so it does not mount `frontend/`.

4. `/media/...` URLs are returned by the API and are origin-relative. The frontend resolves image URLs against `API_BASE`, so media works across origins without extra configuration.

5. If the frontend is hosted on a different origin, allow `GET` and `POST` in your CDN/proxy configuration. No cookies or credentials are used.

## Managed platforms (Render / Railway / Fly.io)

- **Build:** the repository Dockerfile (`python:3.12-slim` with CPU-only PyTorch).
- **Start:** the image runs `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` and defaults to `10000` when `PORT` is unset.
- **Model artifacts:** never bake weights into git or the image. Set `ECOVISION_HF_REPO` to a Hugging Face model repository and the app downloads the checkpoint, optional manifest, and optional metrics at startup. `ECOVISION_MODEL_PATH` overrides the download when you mount a checkpoint directly. If the download fails, the app starts degraded rather than fabricating predictions.
- **Disk:** attach a persistent volume and point `ECOVISION_DATA_DIR` at it. Without a disk, SQLite history and uploads are lost on redeploy or restart.
- **Health check:** `/api/health`.
- **Memory:** CPU inference with MobileNetV2 typically needs roughly 400–800 MB resident; allow 1 GB where possible.
- **Render:** [`render.yaml`](../render.yaml) defines a free Docker web service. The free filesystem is ephemeral, so model artifacts are re-downloaded on each boot and runtime data resets.
- **Environment:**

  ```text
  ECOVISION_DEVICE=cpu
  ECOVISION_DATA_DIR=/data
  ECOVISION_MODEL_DIR=/data/models
  ECOVISION_EVALUATION_DIR=/data/evaluation
  ECOVISION_HF_REPO=<your-user>/ecovision-ai
  ECOVISION_HF_FILENAME=mobilenet_v2_final.pt
  ECOVISION_HF_TOKEN=            # only for a private model repository
  ECOVISION_FRONTEND_DIR=/app/frontend
  ECOVISION_SERVE_FRONTEND=true
  ECOVISION_ALLOWED_ORIGINS=https://app.example.com
  ECOVISION_LOG_LEVEL=INFO
  ```

Fly.io needs a volume mount at `/data`; Railway uses a volume with the same variables. Consult each provider's current volume syntax, since mount paths differ.

## Persistence and backups

| Path | Contents | Backup? |
|---|---|---|
| `data/ecovision.db` | Predictions, feedback, metadata | Yes |
| `data/uploads/<id>/` | 512 px original, heatmap, overlay | Optional (derived) |
| `ml/models/` | Checkpoints + manifest | Yes |
| `ml/evaluation/` | Metrics and plots | Yes (can regenerate) |

SQLite runs in WAL mode. Back up with:

```bash
sqlite3 data/ecovision.db ".backup 'backup-$(date +%F).db'"
```

A consistent snapshot requires including the `-wal`/`-shm` files or using the backup command above. Media and database rows should be backed up together.

## Security checklist

- Add authentication and authorization before exposing real user uploads; this project has none.
- Terminate TLS at a reverse proxy or platform load balancer.
- Set an explicit `ECOVISION_ALLOWED_ORIGINS`; never use `*` in production.
- Keep `ECOVISION_MAX_UPLOAD_MB` small (default 8) and enforce the same limit at the proxy.
- Rate-limit `/api/predict` at the proxy (for example Nginx `limit_req`, Cloudflare rules, or the platform's rate limiter). Inference concurrency is capped in-process by `ECOVISION_MAX_CONCURRENT_INFERENCE`, but that is not a substitute for request rate limiting.
- Run the container as non-root (the provided image does).
- Do not expose `data/` or `ml/` directories directly; media is served through `/media`.
- Review retention policy: uploads are reduced to a 512 px derivative, but they are still user content.
- Pin dependencies and rebuild regularly for security updates.

## Scaling notes

SQLite and local media are deliberate choices for a demo/single-instance deployment. For multi-instance production:

1. Move metadata to PostgreSQL and media to object storage (S3/GCS) with signed URLs.
2. Replace the in-process model with a dedicated inference service or run one replica per instance with a shared model artifact.
3. Move the inference semaphore to a queue (Celery/RQ/SQS) if bursts matter.
4. Add authentication, per-user rate limits, and an audit trail for feedback review.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `/api/predict` returns `503 MODEL_UNAVAILABLE` | No checkpoint resolved | Install artifacts, check `ECOVISION_MODEL_DIR`, restart |
| `/api/health` says `"database": "error"` | Unwritable DB path | Ensure the volume exists and is writable by the runtime user |
| Predictions are slow on first request | CPU thread warm-up / model load | Send a warm-up request after deploy; raise `ECOVISION_MAX_CONCURRENT_INFERENCE` only if CPU allows |
| Grad-CAM images 404 | `data/uploads` not persisted or wrong `ECOVISION_UPLOAD_DIR` | Persist `data/`, check mount permissions |
| CORS errors in the browser | Frontend origin not allowed | Set `ECOVISION_ALLOWED_ORIGINS` and restart |
| Frontend loads from cache after an update | Browser caching static assets | The app sends `Cache-Control: no-cache` for HTML/JS/CSS; hard-refresh once or version assets at the CDN |
| Container health check fails during startup | Model load slower than `start-period` | Increase `HEALTHCHECK --start-period` or reduce model size |
