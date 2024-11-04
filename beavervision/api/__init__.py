# beavervision/api/__init__.py
from fastapi import APIRouter
from . import endpoints

router = APIRouter()
router.include_router(endpoints.router)

# beavervision/core/__init__.py
from .wav2lip_interface import Wav2LipPredictor
from .tts_interface import TextToSpeech
from .face_enhancer import FaceExpressionEnhancer

__all__ = [
    'Wav2LipPredictor',
    'TextToSpeech',
    'FaceExpressionEnhancer'
]