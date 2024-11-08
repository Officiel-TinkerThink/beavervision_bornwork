import os
import torch
import gdown
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define model paths
MODELS_DIR = Path("models")
TTS_MODEL_PATH = MODELS_DIR / "tts_model.pth"
WAV2LIP_MODEL_PATH = MODELS_DIR / "wav2lip_gan.pth"

# PyTorch Models files
MODEL_URLS = {
    "tts_model": "https://huggingface.co/coqui/XTTS-v2/blob/main/model.pth",
    "wav2lip": "https://huggingface.co/spaces/capstonedubtrack/Indiclanguagedubbing/resolve/416598a2eefa2f1b02bea859bda45f18208a53cb/wav2lip_gan.pth"
}

def download_file(url: str, destination: Path, desc: str):
    """Download a file with progress bar."""
    try:
        if destination.exists():
            logger.info(f"{desc} already exists at {destination}")
            return True

        logger.info(f"Downloading {desc} to {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Use gdown for PyTorch Models files, requests for direct downloads
        if "drive.google.com" in url:
            gdown.download(url, str(destination), quiet=False)
        else:
            import requests
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f, tqdm(
                desc=desc,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
                    
        return True
    except Exception as e:
        logger.error(f"Error downloading {desc}: {str(e)}")
        return False

def setup_models():
    """Download and set up all required models."""
    success = True
    
    # Create models directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download TTS model
    if not download_file(
        MODEL_URLS["tts_model"],
        TTS_MODEL_PATH,
        "TTS Model"
    ):
        success = False
    
    # Download Wav2Lip model
    if not download_file(
        MODEL_URLS["wav2lip"],
        WAV2LIP_MODEL_PATH,
        "Wav2Lip Model"
    ):
        success = False
    
    return success

if __name__ == "__main__":
    if setup_models():
        logger.info("All models downloaded successfully!")
    else:
        logger.error("Some models failed to download. Please check the logs.")