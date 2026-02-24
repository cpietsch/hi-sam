# Hi-SAM FastAPI Service

FastAPI service with Docker for [Hi-SAM](https://github.com/ymy-k/Hi-SAM) (Hierarchical Text Segmentation with SAM).

Supports text stroke segmentation and hierarchical (word/line/paragraph) segmentation with point prompts.

## Quick Start

### 1. Download Model Checkpoints

Download the desired model weights from [HuggingFace](https://huggingface.co/GoGiants1/Hi-SAM/tree/main) into a local `models/` directory:

```bash
mkdir -p models

# Using the download helper (with docker)
docker compose --profile download run download-models

# Or manually download (e.g. vit_l variant):
# Stroke segmentation model
wget -P models/ https://huggingface.co/GoGiants1/Hi-SAM/resolve/main/sam_tss_l_hiertext.pth
# Hierarchical detection model
wget -P models/ https://huggingface.co/GoGiants1/Hi-SAM/resolve/main/hi_sam_l.pth
```

Available model variants:

| Variant | Stroke Checkpoint | Hierarchical Checkpoint | Size |
|---------|-------------------|------------------------|------|
| `vit_b` | `sam_tss_b_hiertext.pth` (50 MB) | `hi_sam_b.pth` (67 MB) | Small |
| `vit_l` | `sam_tss_l_hiertext.pth` (123 MB) | `hi_sam_l.pth` (140 MB) | Medium |
| `vit_h` | `sam_tss_h_hiertext.pth` (232 MB) | `hi_sam_h.pth` (249 MB) | Large |

### 2. Build and Run

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`.

**Requirements:** Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support.

### CPU-Only Mode

The service automatically falls back to CPU if no GPU is available. Remove the `deploy.resources` section from `docker-compose.yml` if you don't have a GPU:

```yaml
services:
  hi-sam:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models
```

## API Endpoints

### `GET /health`

Health check.

```bash
curl http://localhost:8000/health
```

### `GET /models`

List available and loaded models.

```bash
curl http://localhost:8000/models
```

### `POST /predict/stroke`

Text stroke segmentation. Returns a binary PNG mask.

```bash
curl -X POST http://localhost:8000/predict/stroke \
  -F "image=@photo.jpg" \
  -F "model_type=vit_l" \
  -F "patch_mode=false" \
  --output stroke_mask.png
```

Parameters:
- `image` (file, required): Input image
- `model_type` (string, optional): `vit_b`, `vit_l`, or `vit_h` (default: `vit_l`)
- `patch_mode` (bool, optional): Use sliding window for better small-text detection (default: `false`)

### `POST /predict/hierarchical`

Hierarchical text segmentation with point prompts. Returns a ZIP archive containing masks for each hierarchy level.

```bash
curl -X POST http://localhost:8000/predict/hierarchical \
  -F "image=@photo.jpg" \
  -F "model_type=vit_l" \
  -F 'points=[[125,275],[200,300]]' \
  --output results.zip
```

Parameters:
- `image` (file, required): Input image
- `model_type` (string, optional): `vit_b`, `vit_l`, or `vit_h` (default: `vit_l`)
- `points` (string, required): JSON array of `[x, y]` coordinates

The returned ZIP contains:
- `stroke_mask.png` - Overall text stroke mask
- `point_N_line.png` - Text-line mask per point
- `point_N_para.png` - Paragraph mask per point
- `point_N_word.png` - Word mask per point
- `metadata.json` - IoU scores and mask shapes

### Interactive Docs

Swagger UI is available at `http://localhost:8000/docs`.

## Project Structure

```
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── app/
│   ├── main.py          # FastAPI endpoints
│   ├── model.py          # Model loading and inference
│   └── schemas.py        # Request/response schemas
├── scripts/
│   └── download_models.py  # HuggingFace model downloader
└── models/               # Model checkpoints (gitignored)
```

## Model Download Script

The included script can download specific model variants:

```bash
# Download vit_l (stroke + hierarchical)
python scripts/download_models.py --model-type vit_l

# Download only stroke model
python scripts/download_models.py --model-type vit_l --stroke-only

# Download all variants
python scripts/download_models.py --all
```
