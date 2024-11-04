# beavervision/api/__init__.py
from fastapi import APIRouter
from . import endpoints

router = APIRouter()
router.include_router(endpoints.router)

# beavervision/core/__init__.py
from beavervision.core.wav2lip_interface import Wav2LipPredictor
from beavervision.core.tts_interface import TextToSpeech
from beavervision.core.face_enhancer import FaceExpressionEnhancer

__all__ = [
    'Wav2LipPredictor',
    'TextToSpeech',
    'FaceExpressionEnhancer'
]