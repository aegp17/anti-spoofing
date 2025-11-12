"""Anti-spoofing document detection service."""

from .detector import DocumentDetector
from .image_processor import ImageProcessor
from .heuristic_detector import HeuristicDetector
from .ml_classifier import MLClassifier

__version__ = "1.0.0"
__all__ = [
    "DocumentDetector",
    "ImageProcessor",
    "HeuristicDetector",
    "MLClassifier",
]

