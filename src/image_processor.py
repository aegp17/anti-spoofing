"""
Image preprocessing and validation module for document detection.
"""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple


class ImageProcessor:
    """Handles image loading, validation, and preprocessing."""
    
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    SUPPORTED_FORMATS = {'JPEG', 'PNG', 'JPG'}
    TARGET_SIZE = (224, 224)
    
    @staticmethod
    def validate_and_load(image_bytes: bytes) -> Image.Image:
        """
        Validate and load image from bytes.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            PIL Image object in RGB format
            
        Raises:
            ValueError: If image is invalid or unsupported
        """
        if len(image_bytes) > ImageProcessor.MAX_FILE_SIZE:
            raise ValueError("Image exceeds maximum size limit (10MB)")
        
        try:
            image = Image.open(image_bytes if hasattr(image_bytes, 'read') 
                               else __import__('io').BytesIO(image_bytes))
            
            if image.format not in ImageProcessor.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported image format: {image.format}")
            
            return image.convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}")
    
    @staticmethod
    def pil_to_cv2(image_pil: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format (BGR)."""
        image_np = np.array(image_pil)
        return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def get_image_dimensions(image_pil: Image.Image) -> Tuple[int, int]:
        """Get image width and height."""
        return image_pil.size
    
    @staticmethod
    def get_aspect_ratio(image_pil: Image.Image) -> float:
        """Calculate aspect ratio (width / height)."""
        width, height = image_pil.size
        return width / height if height > 0 else 0

