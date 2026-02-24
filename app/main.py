import io
import logging
import zipfile
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

from app.model import HiSamModelManager, predict_hierarchical, predict_stroke
from app.schemas import HealthResponse, ModelType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

manager: HiSamModelManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager
    logger.info("Initializing Hi-SAM model manager...")
    manager = HiSamModelManager()
    logger.info(f"Device: {manager.device}")
    logger.info(f"Available checkpoints: {manager.list_available_checkpoints()}")
    yield
    logger.info("Shutting down Hi-SAM service")


app = FastAPI(
    title="Hi-SAM API",
    description=(
        "FastAPI service for Hi-SAM (Hierarchical Text Segmentation with SAM). "
        "Supports text stroke segmentation and hierarchical "
        "(word/line/paragraph) segmentation with point prompts."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        device=manager.device,
        loaded_models=manager.list_loaded(),
    )


@app.get("/models")
async def list_models():
    """List available and loaded models."""
    return {
        "available_checkpoints": manager.list_available_checkpoints(),
        "loaded_models": manager.list_loaded(),
        "device": manager.device,
    }


@app.post("/predict/stroke")
async def predict_stroke_endpoint(
    image: UploadFile = File(..., description="Input image file"),
    model_type: ModelType = Form(default=ModelType.vit_l, description="Model variant"),
    patch_mode: bool = Form(default=False, description="Use sliding window for small text"),
):
    """
    Text stroke segmentation.

    Upload an image and receive a binary mask (PNG) of detected text strokes.
    The mask is white (255) where text is detected and black (0) elsewhere.
    """
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        mask = predict_stroke(
            manager, img, model_type=model_type.value, patch_mode=patch_mode
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Stroke prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    _, png_bytes = cv2.imencode(".png", mask)
    return Response(content=png_bytes.tobytes(), media_type="image/png")


@app.post("/predict/hierarchical")
async def predict_hierarchical_endpoint(
    image: UploadFile = File(..., description="Input image file"),
    model_type: ModelType = Form(default=ModelType.vit_l, description="Model variant"),
    points: str = Form(
        ...,
        description='JSON list of [x,y] points, e.g. "[[125,275],[200,300]]"',
    ),
):
    """
    Hierarchical text segmentation with point prompts.

    Upload an image and provide point coordinates to get word, line,
    and paragraph level segmentation masks. Returns a ZIP archive containing:
    - stroke_mask.png: overall text stroke mask
    - point_N_line.png: text-line mask for each point
    - point_N_para.png: paragraph mask for each point
    - point_N_word.png: word mask for each point
    - metadata.json: IoU scores and shapes
    """
    import json

    try:
        point_list = json.loads(points)
        if not isinstance(point_list, list) or not all(
            isinstance(p, list) and len(p) == 2 for p in point_list
        ):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail='Invalid points format. Expected JSON like "[[x1,y1],[x2,y2]]"',
        )

    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        result = predict_hierarchical(
            manager, img, points=point_list, model_type=model_type.value
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Hierarchical prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Pack results into a ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Stroke mask
        _, png = cv2.imencode(".png", result["stroke_mask"])
        zf.writestr("stroke_mask.png", png.tobytes())

        # Per-point masks
        for i, hi in enumerate(result["hi_masks"]):
            _, png = cv2.imencode(".png", hi["line"])
            zf.writestr(f"point_{i}_line.png", png.tobytes())
            _, png = cv2.imencode(".png", hi["para"])
            zf.writestr(f"point_{i}_para.png", png.tobytes())
            _, png = cv2.imencode(".png", hi["word"])
            zf.writestr(f"point_{i}_word.png", png.tobytes())

        # Metadata
        metadata = {
            "stroke_mask_shape": result["stroke_mask_shape"],
            "points": result["results"],
        }
        zf.writestr("metadata.json", json.dumps(metadata, indent=2))

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=hi_sam_results.zip"},
    )
