"""
Main detection orchestrator that combines heuristics and ML models.
"""
from PIL import Image
from typing import Dict, Any, Optional
from .heuristic_detector import HeuristicDetector
from .ml_classifier import MLClassifier


class DocumentDetector:
    """Orchestrates detection logic combining heuristics and ML models."""
    
    RESPONSE_DOCUMENT = "id document detect"
    RESPONSE_SELFIE = "is selfie"
    
    def __init__(self, use_ml: bool = True, model_path: Optional[str] = None):
        """
        Initialize detector.
        
        Args:
            use_ml: Whether to use ML model as fallback
            model_path: Optional path to ML model file
        """
        self.heuristic = HeuristicDetector()
        self.ml_classifier = MLClassifier(model_path=model_path) if use_ml else None
    
    def detect(self, image_pil: Image.Image) -> Dict[str, Any]:
        """
        Detect whether image is a document or selfie.
        
        Uses hierarchical decision logic:
        1. Heuristic rules (fast)
        2. ML model fallback (if available)
        
        Args:
            image_pil: PIL Image object
            
        Returns:
            Dictionary with 'response' and optional 'confidence' score
        """
        # Gather heuristic features
        has_rect_shape = self.heuristic.detect_rectangular_shape(image_pil)
        has_text = self.heuristic.detect_text_presence(image_pil)
        has_face, face_count = self.heuristic.detect_face_presence(image_pil)
        has_doc_aspect = self.heuristic.check_document_aspect_ratio(image_pil)
        
        # Decision logic: Heuristic-based rules
        # PRIORITY: Text presence is the strongest indicator for documents
        # (documents have text; selfies don't)
        
        # Rule 1: DOCUMENT if text is present (regardless of face)
        # Text is the strongest indicator that it's a document
        if has_text:
            return {
                "response": self.RESPONSE_DOCUMENT,
                "method": "heuristic_rule_1_text_detected"
            }
        
        # Rule 2: DOCUMENT if rectangular shape + document aspect ratio
        # (Even without detected text, shape is a strong indicator)
        if has_rect_shape and has_doc_aspect:
            return {
                "response": self.RESPONSE_DOCUMENT,
                "method": "heuristic_rule_2_rectangle_aspect"
            }
        
        # Rule 3: SELFIE if face is prominent and NO text
        # (Text absence + prominent face = selfie)
        if has_face and not has_text:
            return {
                "response": self.RESPONSE_SELFIE,
                "method": "heuristic_rule_3_face_no_text"
            }
        
        # Fallback to ML model if available
        if self.ml_classifier and self.ml_classifier.available:
            return self._ml_classification(image_pil)
        
        # Default fallback based on face presence
        if has_face:
            return {
                "response": self.RESPONSE_SELFIE,
                "method": "default_face"
            }
        else:
            return {
                "response": self.RESPONSE_DOCUMENT,
                "method": "default_document"
            }
    
    def _ml_classification(self, image_pil: Image.Image) -> Dict[str, Any]:
        """
        Perform ML-based classification.
        
        Args:
            image_pil: PIL Image object
            
        Returns:
            Classification result with confidence
        """
        score = self.ml_classifier.get_confidence(image_pil)
        
        if score is None:
            # ML failed, return default based on heuristics
            has_face, _ = self.heuristic.detect_face_presence(image_pil)
            return {
                "response": self.RESPONSE_SELFIE if has_face else self.RESPONSE_DOCUMENT,
                "method": "heuristic_fallback"
            }
        
        if score >= self.ml_classifier.DOCUMENT_THRESHOLD:
            return {
                "response": self.RESPONSE_DOCUMENT,
                "confidence": round(score, 4),
                "method": "ml_model"
            }
        else:
            return {
                "response": self.RESPONSE_SELFIE,
                "confidence": round(1 - score, 4),
                "method": "ml_model"
            }

