"""
Test script for local testing of the detector without running the server.
"""
import sys
from pathlib import Path
from PIL import Image
from src.detector import DocumentDetector
from src.image_processor import ImageProcessor


def test_detector(image_path: str) -> None:
    """
    Test detector on a single image.
    
    Args:
        image_path: Path to test image
    """
    try:
        # Load image
        image_path = Path(image_path)
        
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            return
        
        # Load and process
        image = Image.open(image_path).convert("RGB")
        
        # Initialize detector
        detector = DocumentDetector(use_ml=True)
        
        # Run detection
        result = detector.detect(image)
        
        # Print results
        print(f"\n📸 Image: {image_path.name}")
        print(f"📊 Result: {result}")
        print(f"✅ Detection complete")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def batch_test(image_dir: str) -> None:
    """
    Test detector on all images in a directory.
    
    Args:
        image_dir: Path to directory with test images
    """
    from pathlib import Path
    
    image_dir = Path(image_dir)
    
    if not image_dir.exists():
        print(f"❌ Directory not found: {image_dir}")
        return
    
    detector = DocumentDetector(use_ml=True)
    
    supported_formats = {'.jpg', '.jpeg', '.png'}
    image_files = [
        f for f in image_dir.iterdir()
        if f.suffix.lower() in supported_formats
    ]
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return
    
    print(f"\n🔄 Processing {len(image_files)} images...\n")
    
    for image_path in image_files:
        try:
            image = Image.open(image_path).convert("RGB")
            result = detector.detect(image)
            print(f"📸 {image_path.name:30} → {result['response']}")
        except Exception as e:
            print(f"❌ {image_path.name:30} → Error: {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_detector.py <image_path>     - Test single image")
        print("  python test_detector.py --batch <dir>    - Test all images in directory")
        sys.exit(1)
    
    if sys.argv[1] == "--batch" and len(sys.argv) >= 3:
        batch_test(sys.argv[2])
    else:
        test_detector(sys.argv[1])

