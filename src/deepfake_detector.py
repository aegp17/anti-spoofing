"""
Deepfake Detection Module - Minimal Implementation for Testing

MVP Approach:
1. Face detection (Haar Cascade)
2. Heuristic analysis (sharpness, skin texture)
3. Simple scoring
4. Video frame aggregation
"""

import logging
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Tuple
import os

logger = logging.getLogger(__name__)


class DeepfakeDetector:
    """
    Minimal deepfake detector using heuristics.
    No ML dependencies required for MVP testing.
    """
    
    # Response types
    RESPONSE_DEEPFAKE = "likely_deepfake"
    RESPONSE_REAL = "likely_real"
    RESPONSE_NO_FACE = "no_face_detected"
    
    # Configuration
    FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    DEEPFAKE_THRESHOLD = 0.5
    MIN_FACE_SIZE = 30  # minimum face width/height in pixels
    
    # Heuristic thresholds
    SHARPNESS_THRESHOLD_LOW = 50.0
    SHARPNESS_THRESHOLD_HIGH = 300.0
    SKIN_TEXTURE_VARIANCE_THRESHOLD = 500.0
    
    def __init__(self):
        """Initialize the detector."""
        try:
            self.face_cascade = cv2.CascadeClassifier(self.FACE_CASCADE_PATH)
            if self.face_cascade.empty():
                logger.warning("❌ Failed to load Haar Cascade face detector")
                self.face_cascade = None
            else:
                logger.info("✅ Haar Cascade face detector loaded")
        except Exception as e:
            logger.error(f"❌ Error loading face cascade: {e}")
            self.face_cascade = None
    
    def detect_image(self, image_pil: Image.Image) -> Dict[str, Any]:
        """
        Detect deepfake in a single image.
        
        Args:
            image_pil: PIL Image object
            
        Returns:
            Dict with response, confidence, and method
        """
        logger.debug("Starting image deepfake detection")
        
        try:
            # Convert PIL to OpenCV format
            img_np = np.array(image_pil)
            
            # Validate
            if img_np is None or img_np.size == 0:
                logger.warning("❌ Invalid image data")
                return {
                    "response": self.RESPONSE_NO_FACE,
                    "confidence": 0.0,
                    "method": "invalid_image"
                }
            
            # Convert to grayscale for face detection
            if len(img_np.shape) == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np
            
            # Detect faces
            logger.debug("Detecting faces...")
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.MIN_FACE_SIZE, self.MIN_FACE_SIZE)
            )
            
            if len(faces) == 0:
                logger.info("ℹ️  No faces detected in image")
                return {
                    "response": self.RESPONSE_NO_FACE,
                    "confidence": 0.0,
                    "method": "no_face_detected"
                }
            
            logger.debug(f"Found {len(faces)} face(s)")
            
            # Use the largest face
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face
            face_region = img_np[y:y+h, x:x+w]
            
            # Analyze heuristics
            logger.debug("Analyzing heuristics...")
            sharpness = self._analyze_sharpness(face_region)
            skin_variance = self._analyze_skin_texture(face_region)
            
            logger.debug(f"Heuristics: sharpness={sharpness:.2f}, skin_variance={skin_variance:.2f}")
            
            # Score calculation (simple)
            score = self._calculate_score(sharpness, skin_variance)
            
            # Decision
            if score >= self.DEEPFAKE_THRESHOLD:
                response = self.RESPONSE_DEEPFAKE
                confidence = score
                logger.info(f"✓ Deepfake detected: confidence={confidence:.2%}")
            else:
                response = self.RESPONSE_REAL
                confidence = 1.0 - score
                logger.info(f"✓ Real image detected: confidence={confidence:.2%}")
            
            return {
                "response": response,
                "confidence": round(float(confidence), 4),
                "method": "heuristic_analysis",
                "details": {
                    "sharpness": round(float(sharpness), 2),
                    "skin_variance": round(float(skin_variance), 2),
                    "faces_detected": len(faces)
                }
            }
        
        except Exception as e:
            logger.error(f"❌ Error in image detection: {e}")
            return {
                "response": self.RESPONSE_NO_FACE,
                "confidence": 0.0,
                "method": "error",
                "error": str(e)
            }
    
    def detect_video(
        self,
        video_path: str,
        frame_step: int = 10,
        max_frames: int = 50
    ) -> Dict[str, Any]:
        """
        Detect deepfakes in video by sampling frames.
        
        Args:
            video_path: Path to video file
            frame_step: Analyze every N-th frame
            max_frames: Maximum frames to analyze
            
        Returns:
            Dict with aggregated results
        """
        logger.debug(f"Starting video detection: {video_path}")
        
        try:
            if not os.path.exists(video_path):
                logger.error(f"❌ Video file not found: {video_path}")
                raise ValueError(f"Video not found: {video_path}")
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"❌ Cannot open video: {video_path}")
                raise ValueError(f"Cannot open video: {video_path}")
            
            frame_idx = 0
            scores = []
            frames_analyzed = 0
            
            logger.debug(f"Sampling video: every {frame_step} frames, max {max_frames}")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames
                if frame_idx % frame_step == 0 and frames_analyzed < max_frames:
                    try:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_pil = Image.fromarray(frame_rgb)
                        
                        # Analyze frame
                        result = self.detect_image(frame_pil)
                        
                        if result["response"] != self.RESPONSE_NO_FACE:
                            scores.append(result["confidence"])
                            frames_analyzed += 1
                            logger.debug(f"Frame {frame_idx}: {result['response']} ({result['confidence']:.2%})")
                    
                    except Exception as e:
                        logger.warning(f"⚠️  Error processing frame {frame_idx}: {e}")
                
                frame_idx += 1
            
            cap.release()
            
            # Aggregate results
            if not scores:
                logger.info("ℹ️  No analyzable frames in video")
                return {
                    "response": self.RESPONSE_NO_FACE,
                    "confidence_mean": 0.0,
                    "confidence_max": 0.0,
                    "confidence_min": 0.0,
                    "frames_analyzed": 0,
                    "method": "no_face_in_video"
                }
            
            mean_score = np.mean(scores)
            max_score = np.max(scores)
            min_score = np.min(scores)
            
            logger.debug(f"Video scores: mean={mean_score:.2%}, max={max_score:.2%}, min={min_score:.2%}")
            
            # Decision based on mean score
            if mean_score >= self.DEEPFAKE_THRESHOLD:
                response = self.RESPONSE_DEEPFAKE
                logger.info(f"✓ Deepfake video detected: mean={mean_score:.2%}")
            else:
                response = self.RESPONSE_REAL
                logger.info(f"✓ Real video detected: mean={mean_score:.2%}")
            
            return {
                "response": response,
                "confidence_mean": round(float(mean_score), 4),
                "confidence_max": round(float(max_score), 4),
                "confidence_min": round(float(min_score), 4),
                "frames_analyzed": len(scores),
                "total_frames_sampled": frames_analyzed,
                "method": "heuristic_video_aggregation"
            }
        
        except Exception as e:
            logger.error(f"❌ Error in video detection: {e}")
            return {
                "response": self.RESPONSE_NO_FACE,
                "confidence_mean": 0.0,
                "frames_analyzed": 0,
                "method": "error",
                "error": str(e)
            }
    
    def _analyze_sharpness(self, image_region: np.ndarray) -> float:
        """
        Calculate sharpness using Laplacian variance.
        Higher = sharper
        """
        try:
            if len(image_region.shape) == 3:
                gray = cv2.cvtColor(image_region, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_region
            
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            
            return float(variance)
        except Exception as e:
            logger.warning(f"⚠️  Error calculating sharpness: {e}")
            return 100.0  # Default value
    
    def _analyze_skin_texture(self, image_region: np.ndarray) -> float:
        """
        Analyze skin texture variance in HSV space.
        Measures texture smoothness.
        """
        try:
            if len(image_region.shape) == 3:
                # Convert to HSV
                if image_region.shape[2] == 3:
                    if image_region.dtype == np.uint8:
                        hsv = cv2.cvtColor(image_region, cv2.COLOR_RGB2HSV)
                    else:
                        # Likely already normalized
                        hsv = cv2.cvtColor((image_region * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
                else:
                    hsv = image_region
                
                # Use Value channel for texture analysis
                v_channel = hsv[:, :, 2]
                
                # Calculate local variance
                variance = v_channel.var()
                
                return float(variance)
            else:
                return 500.0  # Default for grayscale
        
        except Exception as e:
            logger.warning(f"⚠️  Error analyzing skin texture: {e}")
            return 500.0  # Default value
    
    def _calculate_score(self, sharpness: float, skin_variance: float) -> float:
        """
        Calculate deepfake probability score (0.0 to 1.0).
        
        Simple heuristic:
        - Too sharp or too blurry = likely fake
        - Low skin texture variance = likely fake (too smooth)
        - Normal sharpness + normal texture = likely real
        """
        score = 0.0
        
        # Sharpness check
        if sharpness < self.SHARPNESS_THRESHOLD_LOW or sharpness > self.SHARPNESS_THRESHOLD_HIGH:
            score += 0.4  # 40% likelihood of being fake
        else:
            score -= 0.2  # Reduce score (more likely real)
        
        # Skin texture check
        if skin_variance < self.SKIN_TEXTURE_VARIANCE_THRESHOLD:
            score += 0.5  # 50% likelihood of being fake (too smooth)
        else:
            score -= 0.2  # Reduce score (more likely real)
        
        # Clamp to [0, 1]
        return float(np.clip(score, 0.0, 1.0))
