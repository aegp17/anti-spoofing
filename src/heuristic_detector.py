"""
Heuristic-based detection module for document vs selfie classification.
"""
import cv2
import numpy as np
import pytesseract
from PIL import Image
from typing import Tuple


class HeuristicDetector:
    """Implements heuristic rules for detecting documents and selfies."""
    
    # Configuration constants
    EDGE_THRESHOLD_LOW = 60
    EDGE_THRESHOLD_HIGH = 180
    APPROX_EPSILON_FACTOR = 0.02
    MIN_CONTOUR_AREA = 8000
    MIN_FACE_AREA_RATIO = 0.3
    MIN_TEXT_LENGTH = 10
    DOCUMENT_ASPECT_RATIO_MIN = 1.3
    DOCUMENT_ASPECT_RATIO_MAX = 1.0 / 1.3
    
    @staticmethod
    def detect_rectangular_shape(image_pil: Image.Image) -> bool:
        """
        Detect if image contains a clear rectangular shape (characteristic of documents).
        
        Args:
            image_pil: PIL Image object
            
        Returns:
            True if rectangular shape detected, False otherwise
        """
        img_np = np.array(image_pil)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(
            gray,
            HeuristicDetector.EDGE_THRESHOLD_LOW,
            HeuristicDetector.EDGE_THRESHOLD_HIGH
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Look for quadrilateral shapes with sufficient area
        for contour in contours:
            approx = cv2.approxPolyDP(
                contour,
                HeuristicDetector.APPROX_EPSILON_FACTOR * cv2.arcLength(contour, True),
                True
            )
            
            area = cv2.contourArea(contour)
            
            if len(approx) == 4 and area > HeuristicDetector.MIN_CONTOUR_AREA:
                return True
        
        return False
    
    @staticmethod
    def detect_text_presence(image_pil: Image.Image) -> bool:
        """
        Detect if image contains significant amount of text (characteristic of documents).
        
        Args:
            image_pil: PIL Image object
            
        Returns:
            True if text detected, False otherwise
        """
        try:
            text = pytesseract.image_to_string(
                image_pil,
                config="--psm 6"
            )
            return len(text.strip()) > HeuristicDetector.MIN_TEXT_LENGTH
        except Exception as e:
            print(f"Error in text detection: {str(e)}")
            return False
    
    @staticmethod
    def detect_face_presence(image_pil: Image.Image) -> Tuple[bool, int]:
        """
        Detect if image contains a human face using Haar Cascade classifier.
        
        Args:
            image_pil: PIL Image object
            
        Returns:
            Tuple of (face_detected: bool, face_count: int)
        """
        img_np = np.array(image_pil)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Load pre-trained Haar Cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        h, w = gray.shape
        image_area = h * w
        
        # Check if face occupies significant portion of image
        for (x, y, fw, fh) in faces:
            face_area = fw * fh
            if face_area > HeuristicDetector.MIN_FACE_AREA_RATIO * image_area:
                return True, len(faces)
        
        return len(faces) > 0, len(faces)
    
    @staticmethod
    def check_document_aspect_ratio(image_pil: Image.Image) -> bool:
        """
        Check if image aspect ratio is consistent with a document (rectangular/card-like).
        
        Args:
            image_pil: PIL Image object
            
        Returns:
            True if aspect ratio matches document, False otherwise
        """
        width, height = image_pil.size
        aspect_ratio = width / height if height > 0 else 0
        
        # Documents are typically wider than tall or have specific aspect ratios
        return (HeuristicDetector.DOCUMENT_ASPECT_RATIO_MAX <= aspect_ratio <= 
                HeuristicDetector.DOCUMENT_ASPECT_RATIO_MIN)

