"""
Enhanced Anti-spoofing detector with improved heuristics.
Phase 1 improvements: +10-15% accuracy gain expected.

Improvements:
1. Enhanced FFT analysis with frequency distribution
2. JPEG artifact detection
3. Color channel consistency analysis
4. Better threshold tuning
"""
import logging
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class EnhancedAntiSpoofingDetector:
    """Enhanced anti-spoofing with Phase 1 improvements."""
    
    RESPONSE_REAL = "selfie"
    RESPONSE_FAKE = "spoofed"
    RESPONSE_NO_FACE = "no_face_detected"
    
    # Improved thresholds
    MIN_FACE_AREA_RATIO = 0.15
    MIN_FACE_HEIGHT = 30
    
    def __init__(self):
        """Initialize enhanced anti-spoofing detector."""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            logger.error(f"❌ Failed to load face cascade")
        else:
            logger.info("✅ Enhanced Face cascade loaded")
    
    def detect(self, image_pil: Image.Image) -> Dict[str, Any]:
        """
        Enhanced anti-spoofing detection with Phase 1 improvements.
        
        Args:
            image_pil: PIL Image object
            
        Returns:
            Dictionary with response, confidence, method, details
        """
        logger.debug("Starting enhanced anti-spoofing detection")
        
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
        
        # Get largest face
        face_bbox = self._get_largest_face(faces, img_bgr)
        x, y, w, h = face_bbox
        face_region = img_bgr[y:y+h, x:x+w]
        
        # Enhanced analysis
        analysis = self._enhanced_analysis(face_region, image_pil)
        
        # Compute final score
        confidence = self._compute_final_score(analysis)
        # More conservative threshold: 0.55 instead of 0.5 to reduce false positives
        response = self.RESPONSE_REAL if confidence < 0.55 else self.RESPONSE_FAKE
        
        logger.info(
            f"Enhanced detection complete: {response} "
            f"(confidence: {confidence:.3f}, method: {analysis['method']})"
        )
        
        return {
            "response": response,
            "confidence": round(confidence, 4),
            "method": analysis["method"],
            "details": {
                "sharpness": round(analysis.get("sharpness", 0), 2),
                "skin_texture": round(analysis.get("skin_texture", 0), 2),
                "fft_entropy": round(analysis.get("fft_entropy", 0), 4),
                "jpeg_artifacts": round(analysis.get("jpeg_artifacts", 0), 4),
                "color_consistency": round(analysis.get("color_consistency", 0), 4)
            }
        }
    
    def _detect_faces(self, img_bgr: np.ndarray) -> list:
        """Detect faces using Haar Cascade."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(self.MIN_FACE_HEIGHT, self.MIN_FACE_HEIGHT)
        )
        return faces.tolist() if len(faces) > 0 else []
    
    def _get_largest_face(self, faces: list, img_bgr: np.ndarray) -> Tuple[int, int, int, int]:
        """Get the largest face bounding box."""
        face_areas = [(f, f[2] * f[3]) for f in faces]
        largest_face, _ = max(face_areas, key=lambda x: x[1])
        return largest_face
    
    def _enhanced_analysis(self, face_region: np.ndarray, image_pil: Image.Image) -> Dict[str, Any]:
        """
        Perform enhanced analysis with Phase 1 improvements.
        """
        # Classic analysis
        sharpness = self._compute_sharpness(face_region)
        skin_texture = self._compute_skin_texture(face_region)
        
        # Phase 1 improvements
        fft_entropy, freq_dist = self._improved_fft_analysis(face_region)
        jpeg_artifacts = self._detect_jpeg_artifacts(image_pil)
        color_consistency = self._analyze_color_consistency(face_region)
        
        # Determine method based on strongest signal
        methods = {
            "sharpness": sharpness,
            "skin_texture": skin_texture,
            "fft_entropy": fft_entropy,
            "jpeg_artifacts": jpeg_artifacts,
            "color_consistency": color_consistency
        }
        
        strongest_signal = max(methods.items(), key=lambda x: abs(x[1] - 0.5))
        method = f"enhanced_{strongest_signal[0]}"
        
        return {
            "sharpness": sharpness,
            "skin_texture": skin_texture,
            "fft_entropy": fft_entropy,
            "jpeg_artifacts": jpeg_artifacts,
            "color_consistency": color_consistency,
            "method": method
        }
    
    def _compute_sharpness(self, face_region: np.ndarray) -> float:
        """Compute sharpness using Laplacian variance."""
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        logger.debug(f"Sharpness: {sharpness:.2f}")
        return sharpness
    
    def _compute_skin_texture(self, face_region: np.ndarray) -> float:
        """Compute skin texture using HSV variance."""
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        sat_variance = np.var(hsv[:, :, 1])
        val_variance = np.var(hsv[:, :, 2])
        skin_texture = sat_variance + val_variance
        
        logger.debug(f"Skin texture: {skin_texture:.2f}")
        return skin_texture
    
    def _improved_fft_analysis(self, face_region: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        PHASE 1: Enhanced FFT analysis with frequency distribution.
        """
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # FFT 2D
            fft = np.fft.fft2(gray)
            magnitude = np.abs(np.fft.fftshift(fft))
            log_magnitude = np.log1p(magnitude)
            
            # Radial frequency analysis
            h, w = log_magnitude.shape
            center_x, center_y = h // 2, w // 2
            
            Y, X = np.ogrid[:h, :w]
            radial_distance = np.sqrt((X - center_y)**2 + (Y - center_x)**2)
            
            # Frequency bands
            low_freq = np.sum(log_magnitude[radial_distance < 30])
            mid_freq = np.sum(log_magnitude[(radial_distance >= 30) & (radial_distance < 60)])
            high_freq = np.sum(log_magnitude[radial_distance >= 60])
            
            total_energy = low_freq + mid_freq + high_freq
            
            # Entropy of frequency distribution
            freq_dist = np.array([low_freq, mid_freq, high_freq]) / (total_energy + 1e-8)
            entropy = -np.sum(freq_dist * np.log(freq_dist + 1e-8))
            
            # Normalize entropy (0-1)
            normalized_entropy = entropy / np.log(3)  # Max entropy for 3 bins
            
            logger.debug(f"FFT Entropy: {normalized_entropy:.4f}")
            return normalized_entropy, freq_dist
        
        except Exception as e:
            logger.warning(f"FFT analysis failed: {str(e)}")
            return 0.5, np.array([0.33, 0.33, 0.33])
    
    def _detect_jpeg_artifacts(self, image_pil: Image.Image) -> float:
        """
        PHASE 1: Detect JPEG compression artifacts (optimized).
        Real photos have different compression patterns than deepfakes.
        """
        try:
            # Sample only center region for speed (most important area)
            img_array = np.array(image_pil)
            h, w = img_array.shape[:2]
            
            # Sample center 200x200 region
            center_h, center_w = h // 2, w // 2
            sample_size = min(200, h, w)
            start_h = max(0, center_h - sample_size // 2)
            start_w = max(0, center_w - sample_size // 2)
            
            gray = cv2.cvtColor(img_array[start_h:start_h+sample_size, start_w:start_w+sample_size], 
                              cv2.COLOR_RGB2GRAY)
            
            # DCT analysis on sampled region (fewer blocks)
            block_size = 8
            total_blocks = 0
            zero_coefficient_ratio = 0
            
            # Sample every 2nd block for speed
            for i in range(0, gray.shape[0] - block_size, block_size * 2):
                for j in range(0, gray.shape[1] - block_size, block_size * 2):
                    block = gray[i:i+block_size, j:j+block_size].astype(np.float32)
                    if block.shape == (block_size, block_size):
                        dct_block = cv2.dct(block / 255.0)
                        zero_count = np.sum(np.abs(dct_block) < 1e-5)
                        zero_coefficient_ratio += zero_count / (block_size * block_size)
                        total_blocks += 1
            
            if total_blocks == 0:
                return 0.5
            
            avg_zero_ratio = zero_coefficient_ratio / total_blocks
            
            # Normalize: Real photos typically 0.3-0.5, outside = suspicious
            if 0.3 <= avg_zero_ratio <= 0.5:
                jpeg_score = 0.2  # Real
            elif avg_zero_ratio < 0.2 or avg_zero_ratio > 0.7:
                jpeg_score = 0.8  # Suspicious
            else:
                jpeg_score = 0.5  # Uncertain
            
            logger.debug(f"JPEG Zero Ratio: {avg_zero_ratio:.4f} → score: {jpeg_score:.2f}")
            return jpeg_score
        
        except Exception as e:
            logger.warning(f"JPEG artifact detection failed: {str(e)}")
            return 0.5
    
    def _analyze_color_consistency(self, face_region: np.ndarray) -> float:
        """
        PHASE 1: Analyze RGB channel consistency.
        Real faces have natural channel correlation; deepfakes may have anomalies.
        """
        try:
            b, g, r = cv2.split(face_region)
            
            # Normalize channels
            r_norm = r.astype(np.float32) / 255.0
            g_norm = g.astype(np.float32) / 255.0
            b_norm = b.astype(np.float32) / 255.0
            
            # Compute cross-channel correlations
            rg_corr = np.corrcoef(r_norm.flatten(), g_norm.flatten())[0, 1]
            rb_corr = np.corrcoef(r_norm.flatten(), b_norm.flatten())[0, 1]
            gb_corr = np.corrcoef(g_norm.flatten(), b_norm.flatten())[0, 1]
            
            # Handle NaN
            rg_corr = rg_corr if not np.isnan(rg_corr) else 0.5
            rb_corr = rb_corr if not np.isnan(rb_corr) else 0.5
            gb_corr = gb_corr if not np.isnan(gb_corr) else 0.5
            
            avg_correlation = (rg_corr + rb_corr + gb_corr) / 3
            
            # Normalize to 0-1 where 1 = inconsistent (likely fake)
            # Real faces typically: correlation > 0.7
            # Deepfakes: correlation < 0.6
            consistency_score = 1 - (avg_correlation + 1) / 2  # Map [-1,1] to [0,1]
            
            logger.debug(f"Color Consistency: {consistency_score:.4f} (avg_corr: {avg_correlation:.3f})")
            return consistency_score
        
        except Exception as e:
            logger.warning(f"Color consistency analysis failed: {str(e)}")
            return 0.5
    
    def _compute_final_score(self, analysis: Dict[str, Any]) -> float:
        """
        Compute final spoofing score combining all Phase 1 metrics.
        
        Score interpretation:
        - 0.0 - 0.4: Likely REAL selfie
        - 0.4 - 0.6: Uncertain
        - 0.6 - 1.0: Likely FAKE/SPOOFED
        """
        sharpness = analysis.get("sharpness", 0)
        skin_texture = analysis.get("skin_texture", 0)
        fft_entropy = analysis.get("fft_entropy", 0.5)
        jpeg_artifacts = analysis.get("jpeg_artifacts", 0.5)
        color_consistency = analysis.get("color_consistency", 0.5)
        
        # Normalize individual scores to 0-1
        # (0 = real, 1 = fake)
        
        # Sharpness: Sharp = real (lower score), Blurry = fake (higher score)
        # More conservative thresholds
        if sharpness > 80:
            sharpness_score = 0.15  # Very sharp = likely real
        elif sharpness > 50:
            sharpness_score = 0.3   # Sharp = probably real
        elif sharpness < 15:
            sharpness_score = 0.8  # Very blurry = likely fake
        else:
            sharpness_score = 0.5  # Medium = uncertain
        
        # Skin texture: Textured = real, Smooth = fake
        if skin_texture > 250:
            skin_texture_score = 0.15  # Textured = likely real
        elif skin_texture > 150:
            skin_texture_score = 0.3   # Some texture = probably real
        elif skin_texture < 80:
            skin_texture_score = 0.8   # Very smooth = likely fake
        else:
            skin_texture_score = 0.5   # Medium = uncertain
        
        # FFT entropy: High entropy = natural (real), Low entropy = artificial (fake)
        if fft_entropy > 0.65:
            fft_score = 0.2   # High entropy = likely real
        elif fft_entropy > 0.45:
            fft_score = 0.4   # Medium-high = probably real
        elif fft_entropy < 0.35:
            fft_score = 0.8   # Low entropy = likely fake
        else:
            fft_score = 0.5   # Medium = uncertain
        
        # JPEG artifacts: Already normalized in _detect_jpeg_artifacts
        jpeg_score = jpeg_artifacts
        
        # Color consistency: More conservative
        if color_consistency < 0.3:
            color_score = 0.2   # Consistent = likely real
        elif color_consistency > 0.7:
            color_score = 0.8   # Inconsistent = likely fake
        else:
            color_score = 0.5   # Medium = uncertain
        
        # Weighted combination (prioritize more reliable metrics)
        weights = {
            "sharpness": 0.25,      # Most reliable
            "skin_texture": 0.25,   # Very reliable
            "fft": 0.20,            # Good signal
            "jpeg": 0.15,           # Less reliable, slower
            "color": 0.15           # Less reliable
        }
        
        final_score = (
            sharpness_score * weights["sharpness"] +
            skin_texture_score * weights["skin_texture"] +
            fft_score * weights["fft"] +
            jpeg_score * weights["jpeg"] +
            color_score * weights["color"]
        )
        
        logger.debug(
            f"Score breakdown: "
            f"sharp={sharpness_score:.2f}, "
            f"texture={skin_texture_score:.2f}, "
            f"fft={fft_score:.2f}, "
            f"jpeg={jpeg_score:.2f}, "
            f"color={color_score:.2f} → "
            f"final={final_score:.3f}"
        )
        
        return final_score

