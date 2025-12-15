"""
Pydantic schemas for API request/response models
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    """Single detection result"""
    label: str = Field(..., description="Detected class name")
    confidence: float = Field(..., description="Detection confidence score", ge=0.0, le=1.0)
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")


class DetectionResponse(BaseModel):
    """Response for single image detection"""
    detections: List[Detection]
    image_shape: List[int] = Field(..., description="[height, width, channels]")
    num_detections: int


class ImageDetectionResult(BaseModel):
    """Detection result for a single image in batch processing"""
    filename: str
    detections: List[Detection]
    num_detections: int
    status: str = "success"
    error: Optional[str] = None


class BatchProcessingResponse(BaseModel):
    """Response for batch image processing"""
    total_images: int
    successful: int
    failed: int
    results: List[ImageDetectionResult]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_path: str
    class_names: List[str]

