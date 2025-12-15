"""
Configuration and path management for the Brain Tumor Detection system
"""
from pathlib import Path

# Get the project root directory (parent of backend/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Model path - relative to project root
MODEL_PATH = PROJECT_ROOT / "Trained_model" / "YOLOv10CM_FYPtrained.pt"

# Dataset path - relative to project root
DATASET_PATH = PROJECT_ROOT / "dataset"

# Class names for brain tumor detection
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Default confidence threshold
DEFAULT_CONFIDENCE = 0.5

def get_model_path() -> Path:
    """Get the model path and verify it exists"""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}\n"
            f"Please ensure the trained model exists in Trained_model/"
        )
    return MODEL_PATH

def get_dataset_path() -> Path:
    """Get the dataset path"""
    return DATASET_PATH

