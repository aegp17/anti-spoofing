"""
🎭 Multi-Purpose Detection Service

Three independent detection modules:

1. 📄 DOCUMENT DETECTION (/detect/document)
   - Distinguishes: ID Document vs Selfie
   - Detects: text, rectangular shapes, card characteristics
   - Response: "id document detect" or "is selfie"

2. 🎭 ANTI-SPOOFING (/detect/antispoofing)
   - Distinguishes: Real Selfie vs Fake/Spoofed
   - Detects: sharpness, skin texture, frequency content
   - Response: "selfie" (real) or "spoofed" (fake)

3. 🤖 DEEPFAKE DETECTION (/analyze/deepfake/image, /analyze/deepfake/video)
   - Distinguishes: Real Face vs Deepfake
   - Detects: facial artifacts, generation patterns
   - Response: "likely_real" or "likely_deepfake"
"""
import io
import logging
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.antispoofing_detector import AntiSpoofingDetector
from src.detector import DocumentDetector
from src.image_processor import ImageProcessor
from src.deepfake_detector import DeepfakeDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Purpose Detection Service",
    description="Document Detection | Anti-Spoofing | Deepfake Detection",
    version="1.0.0"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# INITIALIZATION
# ============================================================================

logger.info("🚀 Initializing detection service...")
antispoofing_detector = AntiSpoofingDetector()
document_detector = DocumentDetector(use_ml=True)
deepfake_detector = DeepfakeDetector()
image_processor = ImageProcessor()
logger.info("✅ Service initialized successfully")


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        JSON with service status
    """
    return JSONResponse({
        "status": "healthy",
        "service": "Multi-Purpose Detection Service",
        "version": "1.0.0",
        "modules": [
            "document_detection",
            "anti_spoofing",
            "deepfake_detection"
        ]
    })


# ============================================================================
# MODULE 1: DOCUMENT DETECTION
# ============================================================================

@app.post("/detect/document")
async def detect_document(file: UploadFile = File(...)):
    """
    📄 DOCUMENT DETECTION MODULE
    
    Detects if image is an ID document or a selfie.
    
    Args:
        file: Uploaded image file (JPEG, PNG)
        
    Returns:
        JSON with detection result:
        - "id document detect": ID document (passport, cedula, etc.)
        - "is selfie": Selfie image
        
    Raises:
        HTTPException: If image is invalid or processing fails
    """
    start_time = time.time()
    logger.info(f"📄 [DOCUMENT] Processing upload: {file.filename}")
    
    try:
        # Read and validate file
        contents = await file.read()
        
        if not contents:
            logger.warning(f"❌ Empty file uploaded: {file.filename}")
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        file_size_mb = len(contents) / (1024 * 1024)
        logger.debug(f"File size: {file_size_mb:.2f}MB")
        
        # Validate and load image
        image = image_processor.validate_and_load(contents)
        logger.debug(f"Image loaded: {image.size[0]}x{image.size[1]} pixels")
        
        # Perform detection
        result = document_detector.detect(image)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ [DOCUMENT] Detection complete: {result['response']} ({elapsed_time:.3f}s)")
        
        return JSONResponse(result)
    
    except ValueError as e:
        logger.error(f"❌ Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"❌ Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ============================================================================
# MODULE 2: ANTI-SPOOFING DETECTION
# ============================================================================

@app.post("/detect/antispoofing")
async def detect_antispoofing(file: UploadFile = File(...)):
    """
    🎭 ANTI-SPOOFING MODULE
    
    Detects if image is a real selfie or a fake/spoofed image.
    
    Args:
        file: Uploaded image file (JPEG, PNG)
        
    Returns:
        JSON with detection result:
        - "selfie": Real selfie (confidence: 0.0-0.5)
        - "spoofed": Fake/spoofed image (confidence: 0.5-1.0)
        - "no_face_detected": No face found in image
        
    Raises:
        HTTPException: If image is invalid or processing fails
    """
    start_time = time.time()
    logger.info(f"🎭 [ANTISPOOFING] Processing upload: {file.filename}")
    
    try:
        # Read and validate file
        contents = await file.read()
        
        if not contents:
            logger.warning(f"❌ Empty file uploaded: {file.filename}")
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        file_size_mb = len(contents) / (1024 * 1024)
        logger.debug(f"File size: {file_size_mb:.2f}MB")
        
        # Validate and load image
        image = image_processor.validate_and_load(contents)
        logger.debug(f"Image loaded: {image.size[0]}x{image.size[1]} pixels")
        
        # Perform anti-spoofing detection
        result = antispoofing_detector.detect(image)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ [ANTISPOOFING] Detection complete: {result['response']} ({elapsed_time:.3f}s)")
        
        return JSONResponse(result)
    
    except ValueError as e:
        logger.error(f"❌ Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"❌ Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ============================================================================
# MODULE 3: DEEPFAKE DETECTION
# ============================================================================

@app.post("/analyze/deepfake/image")
async def detect_deepfake_image(file: UploadFile = File(...)):
    """
    🤖 DEEPFAKE DETECTION MODULE - IMAGE
    
    Detects if image contains a deepfake face or is real.
    
    Args:
        file: Uploaded image file (JPEG, PNG)
        
    Returns:
        JSON with detection result:
        - "likely_real": Real face
        - "likely_deepfake": Deepfake detected
        - "no_face_detected": No face found in image
        - confidence: Float between 0.0 and 1.0
        
    Raises:
        HTTPException: If image is invalid or processing fails
    """
    start_time = time.time()
    logger.info(f"🤖 [DEEPFAKE-IMAGE] Processing upload: {file.filename}")
    
    try:
        # Read and validate file
        contents = await file.read()
        
        if not contents:
            logger.warning(f"❌ Empty file uploaded: {file.filename}")
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        file_size_mb = len(contents) / (1024 * 1024)
        logger.debug(f"File size: {file_size_mb:.2f}MB")
        
        # Validate and load image
        image = image_processor.validate_and_load(contents)
        logger.debug(f"Image loaded: {image.size[0]}x{image.size[1]} pixels")
        
        # Perform deepfake detection
        result = deepfake_detector.detect_image(image)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ [DEEPFAKE-IMAGE] Detection complete: {result['response']} ({elapsed_time:.3f}s)")
        
        return JSONResponse(result)
    
    except ValueError as e:
        logger.error(f"❌ Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"❌ Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/analyze/deepfake/video")
async def detect_deepfake_video(file: UploadFile = File(...)):
    """
    🤖 DEEPFAKE DETECTION MODULE - VIDEO
    
    Detects if video contains deepfake faces or is real.
    
    Args:
        file: Uploaded video file (MP4, AVI, MOV, etc.)
        
    Returns:
        JSON with detection result:
        - "likely_real": Real faces throughout
        - "likely_deepfake": Deepfake detected in frames
        - "no_face_detected_in_video": No faces found
        - confidence_mean: Average confidence across frames
        - frames_evaluated: Number of frames analyzed
        
    Raises:
        HTTPException: If video is invalid or processing fails
    """
    start_time = time.time()
    logger.info(f"🤖 [DEEPFAKE-VIDEO] Processing upload: {file.filename}")
    
    try:
        # Read and validate file
        contents = await file.read()
        
        if not contents:
            logger.warning(f"❌ Empty file uploaded: {file.filename}")
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        file_size_mb = len(contents) / (1024 * 1024)
        logger.debug(f"File size: {file_size_mb:.2f}MB")
        
        # Save video to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(contents)
            video_path = tmp.name
        
        try:
            # Perform deepfake detection
            result = deepfake_detector.detect_video(video_path)
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ [DEEPFAKE-VIDEO] Detection complete: {result['response']} ({elapsed_time:.3f}s)")
            
            return JSONResponse(result)
        
        finally:
            # Clean up temporary file
            import os
            try:
                os.unlink(video_path)
            except:
                pass
    
    except ValueError as e:
        logger.error(f"❌ Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"❌ Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ============================================================================
# LEGACY ENDPOINTS (For backwards compatibility)
# ============================================================================

@app.post("/detect")
async def detect_image_legacy(file: UploadFile = File(...)):
    """
    ⚠️ LEGACY ENDPOINT - Use /detect/antispoofing instead
    
    Detect if uploaded image is a real selfie or spoofed/fake.
    This endpoint is maintained for backwards compatibility.
    
    Args:
        file: Uploaded image file (JPEG, PNG)
        
    Returns:
        JSON with anti-spoofing detection result
    """
    logger.warning("⚠️ Legacy /detect endpoint used. Please use /detect/antispoofing instead.")
    return await detect_antispoofing(file)


@app.post("/detect/batch")
async def detect_batch(files: list[UploadFile] = File(...)):
    """
    Process multiple images in batch for anti-spoofing detection.
    
    Args:
        files: List of image files
        
    Returns:
        JSON array with detection results for each image
    """
    start_time = time.time()
    logger.info(f"🎭 [ANTISPOOFING-BATCH] Processing {len(files)} files")
    results = []
    
    try:
        for idx, file in enumerate(files, 1):
            logger.debug(f"Processing batch item {idx}/{len(files)}: {file.filename}")
            contents = await file.read()
            
            if not contents:
                logger.warning(f"⚠️ Empty file in batch: {file.filename}")
                results.append({"filename": file.filename, "error": "Empty file"})
                continue
            
            try:
                image = image_processor.validate_and_load(io.BytesIO(contents))
                detection = antispoofing_detector.detect(image)
                results.append({"filename": file.filename, **detection})
                logger.debug(f"✓ {file.filename}: {detection['response']}")
            except ValueError as e:
                logger.error(f"❌ {file.filename}: {str(e)}")
                results.append({"filename": file.filename, "error": str(e)})
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ [ANTISPOOFING-BATCH] Processing complete: {len(results)} files ({elapsed_time:.3f}s)")
        
        return JSONResponse({"results": results})
    
    except Exception as e:
        logger.exception(f"❌ Batch processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")
