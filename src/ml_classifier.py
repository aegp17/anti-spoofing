"""
Machine Learning based classification module using MobileNetV2.

Note: PyTorch is optional. Module gracefully handles missing torch installation.
"""
from PIL import Image
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)

# Optional imports
try:
    import torch
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. ML classifier will be disabled.")


class MLClassifier:
    """CNN-based classifier for document vs selfie detection."""
    
    MODEL_PATH = "models/model_mobilenet_v2.pt"
    TARGET_SIZE = (224, 224)
    DOCUMENT_THRESHOLD = 0.85
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize ML classifier.
        
        Args:
            model_path: Optional path to model file. If not provided, uses default path.
        """
        self.model = None
        self.available = False
        self.transform = None
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not installed. ML classifier disabled. Using heuristics only.")
            return
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model_file = model_path or self.MODEL_PATH
        
        if os.path.exists(model_file):
            try:
                self.model = torch.jit.load(model_file, map_location=self.device)
                self.model.eval()
                self.available = True
                logger.info(f"✓ ML model loaded successfully from {model_file}")
            except Exception as e:
                logger.warning(f"⚠ Failed to load ML model: {str(e)}")
                self.available = False
        else:
            logger.info(f"⚠ ML model not found at {model_file}. Using heuristics only.")
            self.available = False
        
        # Define image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize(self.TARGET_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict(self, image_pil: Image.Image) -> Optional[float]:
        """
        Predict probability that image is a document.
        
        Args:
            image_pil: PIL Image object
            
        Returns:
            Probability score (0-1) or None if model unavailable
        """
        if not self.available or self.model is None:
            return None
        
        try:
            # Transform image
            x = self.transform(image_pil).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                logit = self.model(x)
                probability = torch.sigmoid(logit).item()
            
            return probability
        except Exception as e:
            print(f"⚠ Error during ML prediction: {str(e)}")
            return None
    
    def is_document_ml(self, image_pil: Image.Image) -> bool:
        """
        Determine if image is a document based on ML model.
        
        Args:
            image_pil: PIL Image object
            
        Returns:
            True if classified as document, False otherwise
        """
        score = self.predict(image_pil)
        return score is not None and score >= self.DOCUMENT_THRESHOLD
    
    def get_confidence(self, image_pil: Image.Image) -> Optional[float]:
        """
        Get the raw prediction score.
        
        Args:
            image_pil: PIL Image object
            
        Returns:
            Probability score or None if unavailable
        """
        return self.predict(image_pil)

