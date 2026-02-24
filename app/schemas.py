from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ModelType(str, Enum):
    vit_b = "vit_b"
    vit_l = "vit_l"
    vit_h = "vit_h"


class StrokeRequest(BaseModel):
    model_type: ModelType = Field(
        default=ModelType.vit_l,
        description="Model backbone variant: vit_b, vit_l, or vit_h",
    )
    patch_mode: bool = Field(
        default=False,
        description="Use sliding window patches for better small-text quality",
    )


class HierarchicalRequest(BaseModel):
    model_type: ModelType = Field(
        default=ModelType.vit_l,
        description="Model backbone variant: vit_b, vit_l, or vit_h",
    )
    points: list[list[float]] = Field(
        ...,
        description="List of [x, y] point coordinates for prompting",
    )


class HealthResponse(BaseModel):
    status: str
    device: str
    loaded_models: list[str]


class StrokeResponse(BaseModel):
    success: bool
    message: str
    mask_shape: Optional[list[int]] = None


class HierarchicalResponse(BaseModel):
    success: bool
    message: str
    num_points: Optional[int] = None
