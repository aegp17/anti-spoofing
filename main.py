"""
FastAPI application for document vs selfie detection.

Endpoints:
    POST /detect - Detect if uploaded image is a document or selfie
    GET /health - Health check endpoint
"""
import io
import logging
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.detector import DocumentDetector
from src.image_processor import ImageProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Anti-Spoofing Document Detector",
    description="Service to detect document vs selfie images",
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

# Initialize detector
logger.info("Initializing Document Detector...")
detector = DocumentDetector(use_ml=True)
image_processor = ImageProcessor()
logger.info("✓ Service initialized successfully")


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        JSON with service status
    """
    return JSONResponse({
        "status": "healthy",
        "service": "Anti-Spoofing Document Detector",
        "version": "1.0.0"
    })


@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    """
    Detect if uploaded image is a document or selfie.
    
    Args:
        file: Uploaded image file (JPEG, PNG)
        
    Returns:
        JSON with detection result and optional confidence score
        
    Raises:
        HTTPException: If image is invalid or processing fails
    """
    start_time = time.time()
    logger.info(f"📥 Processing upload: {file.filename}")
    
    try:
        # Read file contents
        contents = await file.read()
        
        if not contents:
            logger.warning(f"❌ Empty file uploaded: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
        
        file_size_mb = len(contents) / (1024 * 1024)
        logger.debug(f"File size: {file_size_mb:.2f}MB")
        
        # Validate and load image
        image = image_processor.validate_and_load(contents)
        logger.debug(f"Image loaded: {image.size[0]}x{image.size[1]} pixels")
        
        # Perform detection
        result = detector.detect(image)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Detection complete: {result['response']} ({elapsed_time:.3f}s)")
        
        return JSONResponse(result)
    
    except ValueError as e:
        logger.error(f"❌ Validation error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.exception(f"❌ Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/detect/batch")
async def detect_batch(files: list[UploadFile] = File(...)):
    """
    Process multiple images in batch.
    
    Args:
        files: List of image files
        
    Returns:
        JSON array with detection results for each image
        
    Raises:
        HTTPException: If processing fails
    """
    start_time = time.time()
    logger.info(f"📦 Batch processing started: {len(files)} files")
    results = []
    
    try:
        for idx, file in enumerate(files, 1):
            logger.debug(f"Processing batch item {idx}/{len(files)}: {file.filename}")
            contents = await file.read()
            
            if not contents:
                logger.warning(f"⚠️ Empty file in batch: {file.filename}")
                results.append({
                    "filename": file.filename,
                    "error": "Empty file"
                })
                continue
            
            try:
                image = image_processor.validate_and_load(io.BytesIO(contents))
                detection = detector.detect(image)
                results.append({
                    "filename": file.filename,
                    **detection
                })
                logger.debug(f"✓ {file.filename}: {detection['response']}")
            except ValueError as e:
                logger.error(f"❌ {file.filename}: {str(e)}")
                results.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Batch processing complete: {len(results)} files ({elapsed_time:.3f}s)")
        
        return JSONResponse({"results": results})
    
    except Exception as e:
        logger.exception(f"❌ Batch processing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch processing failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

