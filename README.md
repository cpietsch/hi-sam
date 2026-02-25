# Hi-SAM FastAPI Service

FastAPI service with Docker for [Hi-SAM](https://github.com/ymy-k/Hi-SAM) (Hierarchical Text Segmentation with SAM).

Supports text stroke segmentation and hierarchical (word/line/paragraph) segmentation with point prompts.

## Quick Start

### 1. Download Model Checkpoints

Download the desired model weights from [HuggingFace](https://huggingface.co/GoGiants1/Hi-SAM/tree/main) into a local `models/` directory:

```bash
mkdir -p models

# Using the download helper (with Docker Compose)
docker compose --profile download run download-models

# Or manually download (e.g. vit_l variant):
wget -P models/ https://huggingface.co/GoGiants1/Hi-SAM/resolve/main/sam_tss_l_hiertext.pth
wget -P models/ https://huggingface.co/GoGiants1/Hi-SAM/resolve/main/hi_sam_l.pth
```

Available model variants:

| Variant | Stroke Checkpoint | Hierarchical Checkpoint |
|---------|-------------------|------------------------|
| `vit_b` | `sam_tss_b_hiertext.pth` | `hi_sam_b.pth` |
| `vit_l` | `sam_tss_l_hiertext.pth` | `hi_sam_l.pth` |
| `vit_h` | `sam_tss_h_hiertext.pth` | `hi_sam_h.pth` |

### 2. Build and Run

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`.

**Requirements:** Docker. The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is only needed for GPU deployments.

### CPU-Only Mode

The service automatically falls back to CPU if no GPU is available. For a CPU-only image (no CUDA dependencies), use the CPU compose file:

```bash
docker compose -f docker-compose.cpu.yml up --build
```

To download models with the CPU image:

```bash
docker compose -f docker-compose.cpu.yml --profile download run download-models
```

## API Endpoints

### `GET /health`

Health check. Returns device info and loaded models.

```bash
curl http://localhost:8000/health
```

### `GET /models`

List available checkpoints on disk and currently loaded models.

```bash
curl http://localhost:8000/models
```

### `POST /predict/stroke`

Text stroke segmentation. Returns a binary PNG mask (white = text, black = background).

```bash
curl -X POST http://localhost:8000/predict/stroke \
  -F "image=@photo.jpg" \
  -F "model_type=vit_l" \
  -F "patch_mode=false" \
  --output stroke_mask.png
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image` | file | yes | | Input image |
| `model_type` | string | no | `vit_l` | `vit_b`, `vit_l`, or `vit_h` |
| `patch_mode` | bool | no | `false` | Sliding-window mode for better small-text detection |

### `POST /predict/hierarchical`

Hierarchical text segmentation with point prompts. Returns a ZIP archive containing masks for each hierarchy level.

```bash
curl -X POST http://localhost:8000/predict/hierarchical \
  -F "image=@photo.jpg" \
  -F "model_type=vit_l" \
  -F 'points=[[125,275],[200,300]]' \
  --output results.zip
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image` | file | yes | | Input image |
| `model_type` | string | no | `vit_l` | `vit_b`, `vit_l`, or `vit_h` |
| `points` | string | yes | | JSON array of `[x, y]` coordinates |

The returned ZIP contains:
- `stroke_mask.png` -- overall text stroke mask
- `point_N_word.png` -- word mask per point
- `point_N_line.png` -- text-line mask per point
- `point_N_para.png` -- paragraph mask per point
- `metadata.json` -- IoU scores and mask shapes

### Interactive Docs

Swagger UI is available at `http://localhost:8000/docs`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HISAM_REPO_PATH` | `/opt/Hi-SAM` | Path to the cloned Hi-SAM repository inside the container |
| `MODEL_DIR` | `/models` | Directory containing model checkpoint files |

## Deployment

### Docker Compose

The default `docker-compose.yml` includes two services:

- **hi-sam** -- the main FastAPI server (GPU-enabled by default)
- **download-models** -- one-shot helper that runs `scripts/download_models.py` in the main image (activated via `--profile download`)

### Kubernetes / Custom Platforms

`deployment.yaml` provides a pod spec template with:

- An **init container** (same image as the main container) that runs `scripts/download_models.py` to fetch `vit_l` checkpoints before the main container starts.
- A **main container** (`ghcr.io/cpietsch/hi-sam:latest`) running the FastAPI server with GPU resources.
- A **5 Gi persistent volume** mounted at `/models`.

Adapt the spec to your platform (standard Kubernetes Deployment, Helm chart, etc.) as needed. For CPU-only deployments, build an image with `BASE_IMAGE=pytorch/pytorch:latest` and remove the `nvidia.com/gpu` limits/requests plus `CUDA_VISIBLE_DEVICES`.

## Model Download Script

The `scripts/download_models.py` helper downloads checkpoints directly from HuggingFace:

```bash
# Download vit_l (stroke + hierarchical)
python scripts/download_models.py --model-type vit_l

# Download only stroke model
python scripts/download_models.py --model-type vit_l --stroke-only

# Download only hierarchical model
python scripts/download_models.py --model-type vit_l --hier-only

# Download all variants
python scripts/download_models.py --all
```

## Project Structure

```
├── Dockerfile              # PyTorch 2.1 + CUDA 11.8 base, clones Hi-SAM repo
├── docker-compose.yml      # GPU service + model download helper
├── deployment.yaml         # Kubernetes pod spec template
├── requirements.txt        # FastAPI + Hi-SAM Python dependencies
├── app/
│   ├── main.py             # FastAPI endpoints and ZIP packaging
│   ├── model.py            # Model loading, caching, and inference
│   └── schemas.py          # Pydantic request/response schemas
├── scripts/
│   └── download_models.py  # CLI tool for downloading checkpoints
└── models/                 # Model checkpoints (gitignored)
```
