import torch
import numpy as np
from typing import Optional
from pathlib import Path
from TTS.api import TTS
from ..utils.logger import setup_logger
from ..config import settings

logger = setup_logger(__name__)

class TTSError(Exception):
    """Base exception for TTS-related errors."""
    pass

class TextToSpeech:
    def __init__(self, model_path: Optional[str] = None):
        self.settings = settings
        self.device = torch.device(self.settings.CUDA_DEVICE)
        self.model_path = Path(model_path) if model_path else Path(self.settings.TTS_MODEL_PATH)
        self.model = self._load_model()
        
    def _load_model(self):
        try:
            logger.info(f"Loading TTS model from {self.model_path}")
            
            # Check if model exists
            if not self.model_path.exists():
                raise TTSError(
                    f"TTS model not found at {self.model_path}. "
                    "Please run 'python -m beavervision.scripts.setup_models' first."
                )
            
            # Initialize TTS
            tts = TTS(
                model_name="tts_models/en/ljspeech/glow-tts",
                progress_bar=False,
                gpu=torch.cuda.is_available()
            )
            
            return tts
            
        except Exception as e:
            logger.error(f"Failed to load TTS model: {str(e)}")
            raise TTSError(f"Failed to load TTS model: {str(e)}")

    async def generate_speech(self, text: str) -> str:
        """
        Convert input text to speech with error handling and validation.
        
        Args:
            text (str): Input text to convert to speech
            
        Returns:
            str: Path to generated audio file
            
        Raises:
            TTSError: If speech generation fails
        """
        try:
            logger.info(f"Generating speech for text: {text[:50]}...")
            
            # Input validation
            if not text or len(text.strip()) == 0:
                raise ValueError("Empty text input")
            
            # Create output directory if it doesn't exist
            output_dir = Path("temp")
            output_dir.mkdir(exist_ok=True)
            
            # Generate unique output path
            output_path = output_dir / f"generated_audio_{int(time.time())}.wav"
            
            # Generate speech
            self.model.tts_to_file(
                text=text,
                file_path=str(output_path)
            )
            
            logger.info(f"Speech generated successfully: {output_path}")
            return str(output_path)
                
        except Exception as e:
            logger.error(f"Speech generation failed: {str(e)}")
            raise TTSError(f"Speech generation failed: {str(e)}")