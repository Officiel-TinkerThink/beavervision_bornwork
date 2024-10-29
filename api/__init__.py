# app/api/__init__.py
from fastapi import APIRouter

# Create a router instance that will be imported by main.py
router = APIRouter()

# Import all endpoints to register them with the router
from .endpoints import *

# app/core/__init__.py
from .wav2lip_interface import Wav2LipPredictor
from .tts_interface import TextToSpeech
from .face_enhancer import FaceExpressionEnhancer

__all__ = [
    'Wav2LipPredictor',
    'TextToSpeech',
    'FaceExpressionEnhancer'
]

# Optional: Add version information
__version__ = '1.0.0'

# Optional: Add package metadata
__author__ = 'Your Name'
__description__ = 'Real-time lip synchronization with facial expressions'

# Add any shared constants or configurations
DEFAULT_DEVICE = 'cuda'
DEFAULT_VIDEO_LENGTH = 5  # minimum video length in seconds
DEFAULT_SAMPLE_RATE = 16000  # audio sample rate

# Add any utility functions that might be needed across modules
def get_device():
    """
    Utility function to determine the available device (CPU/GPU)
    """
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'