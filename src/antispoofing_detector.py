"""
Anti-spoofing detector: distinguishes between real selfies and fake/spoofed images.
"""
import logging
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class AntiSpoofingDetector:
    """Detects spoofed/fake selfies using heuristic analysis."""
    
    RESPONSE_REAL = "selfie"  # Real selfie
    RESPONSE_FAKE = "spoofed"  # Fake/spoofed image
    RESPONSE_NO_FACE = "no_face_detected"
    
    # Thresholds
    MIN_FACE_AREA_RATIO = 0.15  # Face must occupy at least 15% of image
    MIN_FACE_HEIGHT = 30  # Minimum face height in pixels
    
    # Sharpness thresholds (Laplacian variance)
    MIN_SHARPNESS_REAL = 50.0  # Real faces tend to be sharper
    MAX_SHARPNESS_FAKE = 20.0  # Very blurry = likely fake
    
    # Skin texture variance (HSV)
    MIN_SKIN_TEXTURE_REAL = 300.0  # Real skin has texture variation
    MAX_SKIN_TEXTURE_FAKE = 100.0  # Flat skin = likely fake
    
    # Frequency domain analysis
    HIGH_FREQ_THRESHOLD_REAL = 0.15  # Real faces have more high-freq content
    
    def __init__(self):
        """Initialize anti-spoofing detector."""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            logger.error(f"❌ Failed to load face cascade from {cascade_path}")
        else:
            logger.info("✅ Face cascade loaded successfully")
    
    def detect(self, image_pil: Image.Image) -> Dict[str, Any]:
        """
        Detect if image is a real or spoofed selfie.
        
        Args:
            image_pil: PIL Image object
            
        Returns:
            Dictionary with 'response', 'confidence', and 'method'
        """
        logger.debug("Starting anti-spoofing detection")
        
        img_np = np.array(image_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces = self._detect_faces(img_bgr)
        
        if not faces:
            logger.info("No face detected in image")
            return {
                "response": self.RESPONSE_NO_FACE,
                "confidence": 0.0,
                "method": "no_face_detected"
            }
        
        # Use the largest face
        face_bbox = self._get_largest_face(faces, img_bgr)
        
        # Extract face region
        x, y, w, h = face_bbox
        face_region = img_bgr[y:y+h, x:x+w]
        
        # Perform analysis
        analysis = self._analyze_face_region(face_region)
        
        # Combine scores
        confidence = self._compute_spoofing_score(analysis)
        
        response = self.RESPONSE_REAL if confidence < 0.5 else self.RESPONSE_FAKE
        
        logger.info(
            f"Anti-spoofing detection complete: {response} "
            f"(confidence: {confidence:.3f}, method: {analysis['method']})"
        )
        
        return {
            "response": response,
            "confidence": round(confidence, 4),
            "method": analysis["method"],
            "details": {
                "sharpness": round(analysis["sharpness"], 2),
                "skin_texture": round(analysis["skin_texture"], 2),
                "high_freq_ratio": round(analysis["high_freq_ratio"], 4)
            }
        }
    
    def _detect_faces(self, img_bgr: np.ndarray) -> list:
        """
        Detect faces in image using Haar Cascade.
        
        Args:
            img_bgr: OpenCV image (BGR format)
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(self.MIN_FACE_HEIGHT, self.MIN_FACE_HEIGHT)
        )
        
        logger.debug(f"Detected {len(faces)} face(s)")
        return faces.tolist() if len(faces) > 0 else []
    
    def _get_largest_face(self, faces: list, img_bgr: np.ndarray) -> Tuple[int, int, int, int]:
        """Get the largest face bounding box."""
        face_areas = [(f, f[2] * f[3]) for f in faces]
        largest_face, _ = max(face_areas, key=lambda x: x[1])
        
        h, w = img_bgr.shape[:2]
        x, y, fw, fh = largest_face
        
        # Check if face is significant enough
        face_area_ratio = (fw * fh) / (h * w)
        logger.debug(f"Face area ratio: {face_area_ratio:.3f}")
        
        return largest_face
    
    def _analyze_face_region(self, face_region: np.ndarray) -> Dict[str, Any]:
        """
        Analyze face region for spoofing indicators.
        
        Args:
            face_region: Face region extracted from image (BGR)
            
        Returns:
            Dictionary with analysis metrics
        """
        # Sharpness analysis
        sharpness = self._compute_sharpness(face_region)
        
        # Skin texture analysis
        skin_texture = self._compute_skin_texture(face_region)
        
        # Frequency domain analysis
        high_freq_ratio = self._compute_frequency_content(face_region)
        
        # Determine method based on most reliable metric
        if sharpness < self.MAX_SHARPNESS_FAKE:
            method = "low_sharpness_detected"
        elif skin_texture < self.MAX_SKIN_TEXTURE_FAKE:
            method = "flat_skin_texture_detected"
        elif high_freq_ratio < 0.05:
            method = "low_frequency_content_detected"
        else:
            method = "combined_analysis"
        
        return {
            "sharpness": sharpness,
            "skin_texture": skin_texture,
            "high_freq_ratio": high_freq_ratio,
            "method": method
        }
    
    def _compute_sharpness(self, face_region: np.ndarray) -> float:
        """
        Compute sharpness using Laplacian variance.
        Real faces are sharper than printed/displayed images.
        """
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        logger.debug(f"Sharpness (Laplacian variance): {sharpness:.2f}")
        return sharpness
    
    def _compute_skin_texture(self, face_region: np.ndarray) -> float:
        """
        Compute skin texture using HSV variance.
        Real skin has more texture variation than flat printed/displayed skin.
        """
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        
        # Analyze saturation and value channels (texture indicators)
        sat_variance = np.var(hsv[:, :, 1])
        val_variance = np.var(hsv[:, :, 2])
        
        skin_texture = sat_variance + val_variance
        
        logger.debug(f"Skin texture (HSV variance): {skin_texture:.2f}")
        return skin_texture
    
    def _compute_frequency_content(self, face_region: np.ndarray) -> float:
        """
        Compute high-frequency content ratio using FFT.
        Real faces have more high-frequency components than flat images.
        """
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Compute FFT
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude_spectrum = np.abs(fft_shift)
            
            # Normalize
            log_magnitude = np.log1p(magnitude_spectrum)
            
            # Split into center (low freq) and edges (high freq)
            h, w = log_magnitude.shape
            h_quarter, w_quarter = h // 4, w // 4
            
            # Center region (low frequency)
            center_region = log_magnitude[h_quarter:3*h_quarter, w_quarter:3*w_quarter]
            center_energy = np.sum(center_region)
            
            # High frequency regions (avoid concatenation issues)
            top_edges = np.sum(log_magnitude[:h_quarter, :])
            bottom_edges = np.sum(log_magnitude[3*h_quarter:, :])
            left_edges = np.sum(log_magnitude[:, :w_quarter])
            right_edges = np.sum(log_magnitude[:, 3*w_quarter:])
            
            edges_energy = top_edges + bottom_edges + left_edges + right_edges
            total_energy = center_energy + edges_energy
            
            high_freq_ratio = edges_energy / (total_energy + 1e-8)
            
            logger.debug(f"High frequency ratio: {high_freq_ratio:.4f}")
            return high_freq_ratio
        except Exception as e:
            logger.warning(f"Frequency analysis failed: {str(e)}, returning 0.5")
            return 0.5
    
    def _compute_spoofing_score(self, analysis: Dict[str, Any]) -> float:
        """
        Compute final spoofing score (0-1).
        0 = likely real, 1 = likely fake/spoofed.
        """
        sharpness = analysis["sharpness"]
        skin_texture = analysis["skin_texture"]
        high_freq_ratio = analysis["high_freq_ratio"]
        
        # Initialize score
        score = 0.5
        
        # Sharpness metric (0 = real, 1 = fake)
        if sharpness > self.MIN_SHARPNESS_REAL:
            sharpness_score = 0.1  # Sharp = real
        elif sharpness > self.MAX_SHARPNESS_FAKE:
            sharpness_score = 0.5  # Medium = uncertain
        else:
            sharpness_score = 0.9  # Blurry = fake
        
        # Skin texture metric (0 = real, 1 = fake)
        if skin_texture > self.MIN_SKIN_TEXTURE_REAL:
            texture_score = 0.1  # Textured = real
        elif skin_texture > self.MAX_SKIN_TEXTURE_FAKE:
            texture_score = 0.5  # Medium = uncertain
        else:
            texture_score = 0.9  # Flat = fake
        
        # Frequency metric (0 = real, 1 = fake)
        if high_freq_ratio > self.HIGH_FREQ_THRESHOLD_REAL:
            freq_score = 0.1  # High freq = real
        elif high_freq_ratio > 0.05:
            freq_score = 0.5  # Medium = uncertain
        else:
            freq_score = 0.9  # Low freq = fake
        
        # Weighted average
        final_score = (sharpness_score * 0.4 + texture_score * 0.4 + freq_score * 0.2)
        
        logger.debug(
            f"Spoofing scores: sharpness={sharpness_score:.2f}, "
            f"texture={texture_score:.2f}, freq={freq_score:.2f}, "
            f"final={final_score:.2f}"
        )
        
        return final_score

