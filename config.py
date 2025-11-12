"""
Configuration module for anti-spoofing detector service.
"""
from pathlib import Path


class Config:
    """Service configuration."""
    
    # FastAPI
    HOST = "0.0.0.0"
    PORT = 8000
    LOG_LEVEL = "info"
    
    # Model
    MODEL_PATH = "models/model_mobilenet_v2.pt"
    USE_ML_MODEL = True
    
    # Image Processing
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    SUPPORTED_FORMATS = {'JPEG', 'PNG', 'JPG'}
    TARGET_SIZE = (224, 224)
    
    # Detection Thresholds
    DOCUMENT_ML_THRESHOLD = 0.85
    MIN_FACE_AREA_RATIO = 0.3
    MIN_TEXT_LENGTH = 10
    MIN_CONTOUR_AREA = 8000
    
    # Edge Detection
    CANNY_THRESHOLD_LOW = 60
    CANNY_THRESHOLD_HIGH = 180
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent
    MODELS_DIR = PROJECT_ROOT / "models"
    
    @classmethod
    def get_model_path(cls) -> Path:
        """Get full path to model file."""
        return cls.PROJECT_ROOT / cls.MODEL_PATH

