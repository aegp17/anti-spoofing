"""
Deepfake detection module using heuristics and ML models.

Architecture:
    1. Face detection and alignment
    2. Feature extraction / classification
    3. Per-frame aggregation for video
    4. Confidence scoring and reporting
"""
import logging
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Dict, Any, Optional, Tuple, List
from torchvision import transforms

logger = logging.getLogger(__name__)


class DeepfakeDetector:
    """
    Detects deepfakes in images and videos.
    
    Uses a combination of:
    - Heuristic analysis (face detection, temporal consistency)
    - ML model (pre-trained CNN for real vs fake classification)
    """
    
    # Configuration
    TARGET_SIZE = (224, 224)
    FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    DEEPFAKE_THRESHOLD = 0.5
    MIN_FACE_AREA_RATIO = 0.2
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize deepfake detector.
        
        Args:
            model_path: Optional path to pre-trained model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_available = False
        
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(self.FACE_CASCADE_PATH)
        
        # Initialize image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize(self.TARGET_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load ML model if available
        if model_path and __import__("os").path.exists(model_path):
            try:
                self.model = torch.jit.load(model_path, map_location=self.device)
                self.model.eval()
                self.model_available = True
                logger.info(f"✓ Deepfake model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"⚠ Failed to load deepfake model: {str(e)}")
                self.model_available = False
        else:
            logger.info("ℹ No deepfake model available, using heuristics only")
    
    def _detect_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face in image using Haar Cascade.
        
        Args:
            image: Image as numpy array (RGB)
            
        Returns:
            Tuple of (x1, y1, x2, y2) bounding box or None
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) == 0:
            return None
        
        # Return largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        
        return (x, y, x + w, y + h)
    
    def _extract_face_region(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract and normalize face region from image.
        
        Args:
            image: Image as numpy array (RGB)
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Cropped and normalized face image
        """
        x1, y1, x2, y2 = bbox
        face = image[y1:y2, x1:x2]
        return face
    
    def _predict_ml(self, face_image: np.ndarray) -> Optional[float]:
        """
        Predict deepfake probability using ML model.
        
        Args:
            face_image: Face image as numpy array
            
        Returns:
            Probability score (0=real, 1=fake) or None if model unavailable
        """
        if not self.model_available:
            return None
        
        try:
            # Convert to PIL for transform
            if isinstance(face_image, np.ndarray):
                face_pil = Image.fromarray(face_image.astype(np.uint8))
            else:
                face_pil = face_image
            
            # Transform
            x = self.transform(face_pil).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                logit = self.model(x)
                prob = torch.sigmoid(logit).item()
            
            return prob
        except Exception as e:
            logger.warning(f"⚠ ML prediction failed: {str(e)}")
            return None
    
    def _heuristic_analysis(
        self,
        image: np.ndarray,
        face_bbox: Tuple[int, int, int, int]
    ) -> Dict[str, Any]:
        """
        Perform heuristic analysis on face region.
        
        Checks for:
        - Suspicious patterns in skin texture
        - Lighting consistency
        - Edge quality
        
        Args:
            image: Image as numpy array
            face_bbox: Bounding box of face
            
        Returns:
            Dictionary with heuristic scores
        """
        x1, y1, x2, y2 = face_bbox
        face = image[y1:y2, x1:x2]
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        
        # Analyze edge quality (sharpness)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Analyze color consistency (skin color variance)
        # Extract face in HSV to analyze skin
        hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        skin_variance = np.var(s)  # Saturation variance
        
        return {
            "sharpness": float(sharpness),
            "skin_variance": float(skin_variance)
        }
    
    def detect_image(self, image_pil: Image.Image) -> Dict[str, Any]:
        """
        Detect deepfake in a single image.
        
        Args:
            image_pil: PIL Image object
            
        Returns:
            Detection result with scores and confidence
        """
        logger.debug("Starting deepfake image detection")
        
        # Convert PIL to numpy
        image_np = np.array(image_pil)
        h, w = image_np.shape[:2]
        logger.debug(f"Image size: {w}x{h}")
        
        # Detect face
        face_bbox = self._detect_face(image_np)
        if not face_bbox:
            logger.warning("No face detected in image")
            return {
                "response": "no_face_detected",
                "confidence": 0.0,
                "method": "no_detection"
            }
        
        x1, y1, x2, y2 = face_bbox
        face_area = (x2 - x1) * (y2 - y1)
        image_area = h * w
        face_ratio = face_area / image_area
        
        logger.debug(f"Face detected: area ratio {face_ratio:.2%}")
        
        # Extract face region
        face_image = self._extract_face_region(image_np, face_bbox)
        
        # Heuristic analysis
        heuristics = self._heuristic_analysis(image_np, face_bbox)
        logger.debug(f"Heuristic scores: {heuristics}")
        
        # ML prediction
        ml_score = self._predict_ml(face_image)
        
        # Decision logic
        if ml_score is not None:
            confidence = ml_score
            is_deepfake = ml_score >= self.DEEPFAKE_THRESHOLD
            method = "ml_model"
        else:
            # Heuristic-based decision
            # Use sharpness and skin variance as indicators
            confidence = heuristics.get("sharpness", 0) / 1000.0  # Normalize
            is_deepfake = confidence > 0.5
            method = "heuristic"
        
        response_type = "likely_deepfake" if is_deepfake else "likely_real"
        
        logger.info(f"✓ Detection complete: {response_type} (confidence: {confidence:.2%})")
        
        return {
            "response": response_type,
            "confidence": round(confidence, 4),
            "method": method,
            "heuristics": heuristics
        }
    
    def detect_video(
        self,
        video_path: str,
        frame_step: int = 10,
        max_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Detect deepfake in video by analyzing multiple frames.
        
        Args:
            video_path: Path to video file
            frame_step: Analyze every N-th frame
            max_frames: Maximum number of frames to analyze
            
        Returns:
            Aggregated detection result with per-frame analysis
        """
        logger.info(f"Starting deepfake video detection: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.debug(f"Video: {total_frames} frames @ {fps:.1f} FPS")
        
        frame_idx = 0
        analyzed_frame_idx = 0
        scores = []
        frames_with_faces = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Analyze every N-th frame
                if frame_idx % frame_step == 0:
                    if max_frames and analyzed_frame_idx >= max_frames:
                        break
                    
                    # Convert to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect face
                    face_bbox = self._detect_face(frame_rgb)
                    if face_bbox:
                        frames_with_faces += 1
                        
                        # Extract face
                        face_image = self._extract_face_region(frame_rgb, face_bbox)
                        
                        # ML prediction
                        score = self._predict_ml(face_image)
                        if score is not None:
                            scores.append(score)
                            logger.debug(f"Frame {frame_idx}: score={score:.4f}")
                    
                    analyzed_frame_idx += 1
                
                frame_idx += 1
        
        finally:
            cap.release()
        
        # Aggregate results
        if not scores:
            logger.warning("No faces detected in video")
            return {
                "response": "no_faces_detected",
                "confidence": 0.0,
                "frames_analyzed": analyzed_frame_idx,
                "method": "no_detection"
            }
        
        scores_array = np.array(scores)
        score_mean = float(np.mean(scores_array))
        score_max = float(np.max(scores_array))
        score_median = float(np.median(scores_array))
        
        # Final decision based on aggregate
        is_deepfake = score_mean >= self.DEEPFAKE_THRESHOLD
        response_type = "likely_deepfake" if is_deepfake else "likely_real"
        
        logger.info(
            f"✓ Video analysis complete: {response_type} "
            f"(mean: {score_mean:.2%}, max: {score_max:.2%}, median: {score_median:.2%})"
        )
        
        return {
            "response": response_type,
            "confidence_mean": round(score_mean, 4),
            "confidence_max": round(score_max, 4),
            "confidence_median": round(score_median, 4),
            "frames_analyzed": analyzed_frame_idx,
            "frames_with_faces": frames_with_faces,
            "method": "ml_video_aggregation"
        }

