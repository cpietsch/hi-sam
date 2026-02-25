import os
import sys
import logging
import shutil
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

HISAM_REPO_PATH = os.environ.get("HISAM_REPO_PATH", "/opt/Hi-SAM")
MODEL_DIR = os.environ.get("MODEL_DIR", "/models")

# Map checkpoint names per model type
STROKE_CHECKPOINTS = {
    "vit_b": "sam_tss_b_hiertext.pth",
    "vit_l": "sam_tss_l_hiertext.pth",
    "vit_h": "sam_tss_h_hiertext.pth",
}

HIER_CHECKPOINTS = {
    "vit_b": "hi_sam_b.pth",
    "vit_l": "hi_sam_l.pth",
    "vit_h": "hi_sam_h.pth",
}

SAM_ENCODER_CHECKPOINTS = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}


def _ensure_hisam_on_path():
    """Add Hi-SAM repo to sys.path so we can import its modules."""
    if HISAM_REPO_PATH not in sys.path:
        sys.path.insert(0, HISAM_REPO_PATH)


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class HiSamModelManager:
    """Manages loading and caching of Hi-SAM models."""

    def __init__(self):
        self.device = _get_device()
        self._predictors: dict[str, object] = {}
        _ensure_hisam_on_path()

    def _build_args(self, model_type: str, checkpoint: str, hier_det: bool) -> SimpleNamespace:
        return SimpleNamespace(
            model_type=model_type,
            checkpoint=checkpoint,
            device=self.device,
            hier_det=hier_det,
            input_size=[1024, 1024],
            attn_layers=1,
            prompt_len=12,
        )

    def _ensure_encoder_checkpoint(self, model_type: str) -> None:
        encoder_ckpt = SAM_ENCODER_CHECKPOINTS[model_type]
        src_path = os.path.join(MODEL_DIR, encoder_ckpt)
        dst_dir = os.path.join(os.getcwd(), "pretrained_checkpoint")
        dst_path = os.path.join(dst_dir, encoder_ckpt)

        if os.path.exists(dst_path):
            return

        if not os.path.exists(src_path):
            raise FileNotFoundError(
                f"SAM encoder checkpoint not found: {src_path}. "
                f"Download it from https://huggingface.co/GoGiants1/Hi-SAM"
            )

        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src_path, dst_path)

    def _load_model(self, model_type: str, hier_det: bool) -> object:
        """Load a Hi-SAM model and return a SamPredictor."""
        from hi_sam.modeling.build import model_registry
        from hi_sam.modeling.predictor import SamPredictor

        self._ensure_encoder_checkpoint(model_type)

        if hier_det:
            ckpt_name = HIER_CHECKPOINTS[model_type]
        else:
            ckpt_name = STROKE_CHECKPOINTS[model_type]

        checkpoint_path = os.path.join(MODEL_DIR, ckpt_name)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                f"Download it from https://huggingface.co/GoGiants1/Hi-SAM"
            )

        args = self._build_args(model_type, checkpoint_path, hier_det)
        logger.info(f"Loading model {model_type} (hier_det={hier_det}) from {checkpoint_path}")
        model = model_registry[model_type](args)
        model.eval()
        model.to(self.device)
        predictor = SamPredictor(model)
        logger.info(f"Model {model_type} (hier_det={hier_det}) loaded on {self.device}")
        return predictor

    def get_predictor(self, model_type: str, hier_det: bool) -> object:
        """Get or load a cached predictor."""
        cache_key = f"{model_type}_{'hier' if hier_det else 'stroke'}"
        if cache_key not in self._predictors:
            self._predictors[cache_key] = self._load_model(model_type, hier_det)
        return self._predictors[cache_key]

    def list_loaded(self) -> list[str]:
        return list(self._predictors.keys())

    def list_available_checkpoints(self) -> dict[str, list[str]]:
        """List which checkpoints are available on disk."""
        available = {"stroke": [], "hierarchical": []}
        for model_type, ckpt in STROKE_CHECKPOINTS.items():
            if os.path.exists(os.path.join(MODEL_DIR, ckpt)):
                available["stroke"].append(model_type)
        for model_type, ckpt in HIER_CHECKPOINTS.items():
            if os.path.exists(os.path.join(MODEL_DIR, ckpt)):
                available["hierarchical"].append(model_type)
        return available


def predict_stroke(
    manager: HiSamModelManager,
    image: np.ndarray,
    model_type: str = "vit_l",
    patch_mode: bool = False,
) -> np.ndarray:
    """Run text stroke segmentation, returning a binary mask."""
    from shapely.geometry import Polygon
    import pyclipper

    predictor = manager.get_predictor(model_type, hier_det=False)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image

    if patch_mode:
        ori_size = image_rgb.shape[:2]
        patch_list, h_slice_list, w_slice_list = _patchify_sliding(image_rgb, 512, 384)
        mask_patches = []
        for patch in patch_list:
            predictor.set_image(patch)
            m, hr_m, score, hr_score = predictor.predict(
                multimask_output=False, return_logits=True
            )
            mask_patches.append(hr_m[0])
        mask = _unpatchify_sliding(mask_patches, h_slice_list, w_slice_list, ori_size)
        mask = mask > predictor.model.mask_threshold
    else:
        predictor.set_image(image_rgb)
        mask, hr_mask, score, hr_score = predictor.predict(multimask_output=False)
        mask = hr_mask

    if len(mask.shape) == 3:
        mask = mask[0]
    return (mask.astype(np.uint8) * 255)


def predict_hierarchical(
    manager: HiSamModelManager,
    image: np.ndarray,
    points: list[list[float]],
    model_type: str = "vit_l",
) -> dict:
    """Run hierarchical text segmentation with point prompts."""
    from shapely.geometry import Polygon as ShapelyPolygon
    import pyclipper

    predictor = manager.get_predictor(model_type, hier_det=True)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image
    predictor.set_image(image_rgb)

    input_points = np.array(points)
    input_labels = np.ones(input_points.shape[0])

    mask, hr_mask, score, hr_score, hi_mask, hi_iou, word_mask = predictor.predict(
        multimask_output=False,
        hier_det=True,
        point_coords=input_points,
        point_labels=input_labels,
    )

    # Process stroke mask
    stroke_mask = hr_mask
    if len(stroke_mask.shape) == 3:
        stroke_mask = stroke_mask[0]
    stroke_mask = (stroke_mask.astype(np.uint8) * 255)

    # Process hierarchical masks per point
    results = []
    for i in range(len(points)):
        line_mask = hi_mask[i][0].astype(np.uint8) * 255
        para_mask = hi_mask[i][1].astype(np.uint8) * 255
        w_mask = word_mask[i][0].astype(np.uint8) * 255

        # Find the word polygon containing the point
        contours, _ = cv2.findContours(w_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        selected_word_mask = np.zeros_like(w_mask)
        for cont in contours:
            epsilon = 0.002 * cv2.arcLength(cont, True)
            approx = cv2.approxPolyDP(cont, epsilon, True)
            pts = approx.reshape((-1, 2))
            if pts.shape[0] < 4:
                continue
            expanded = _unclip(pts)
            if len(expanded) != 1:
                continue
            expanded = expanded[0].astype(np.int32)
            if cv2.pointPolygonTest(expanded, (int(points[i][0]), int(points[i][1])), False) >= 0:
                selected_word_mask = cv2.fillPoly(np.zeros_like(w_mask), [expanded], 255)
                break

        results.append({
            "point": points[i],
            "line_iou": float(hi_iou[i][1]),
            "para_iou": float(hi_iou[i][2]),
            "line_mask_shape": list(line_mask.shape),
            "para_mask_shape": list(para_mask.shape),
            "word_mask_shape": list(selected_word_mask.shape),
        })

    return {
        "stroke_mask_shape": list(stroke_mask.shape),
        "results": results,
        "stroke_mask": stroke_mask,
        "hi_masks": [
            {
                "line": hi_mask[i][0].astype(np.uint8) * 255,
                "para": hi_mask[i][1].astype(np.uint8) * 255,
                "word": word_mask[i][0].astype(np.uint8) * 255,
            }
            for i in range(len(points))
        ],
    }


def _patchify_sliding(image: np.ndarray, patch_size: int = 512, stride: int = 256):
    h, w = image.shape[:2]
    patch_list = []
    h_slice_list = []
    w_slice_list = []
    for j in range(0, h, stride):
        start_h, end_h = j, j + patch_size
        if end_h > h:
            start_h = max(h - patch_size, 0)
            end_h = h
        for i in range(0, w, stride):
            start_w, end_w = i, i + patch_size
            if end_w > w:
                start_w = max(w - patch_size, 0)
                end_w = w
            h_slice = slice(start_h, end_h)
            h_slice_list.append(h_slice)
            w_slice = slice(start_w, end_w)
            w_slice_list.append(w_slice)
            patch_list.append(image[h_slice, w_slice])
    return patch_list, h_slice_list, w_slice_list


def _unpatchify_sliding(patch_list, h_slice_list, w_slice_list, ori_size):
    whole_logits = np.zeros(ori_size)
    for idx in range(len(patch_list)):
        whole_logits[h_slice_list[idx], w_slice_list[idx]] += patch_list[idx]
    return whole_logits


def _unclip(p, unclip_ratio=2.0):
    from shapely.geometry import Polygon as ShapelyPolygon
    import pyclipper

    poly = ShapelyPolygon(p)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(p, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded
