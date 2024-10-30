from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import tempfile
from pathlib import Path
from beavervision.core.wav2lip_interface import Wav2LipPredictor
from beavervision.core.tts_interface import TextToSpeech, TTSError
from beavervision.core.face_enhancer import FaceExpressionEnhancer
from beavervision.utils.validators import validate_video
from beavervision.utils.logger import setup_logger
from beavervision.utils.monitoring import REQUESTS_TOTAL, monitor_timing

router = APIRouter()
logger = setup_logger(__name__)

class LipSyncPipeline:
    def __init__(self):
        self.settings = Settings()
        self.device = torch.device(self.settings.CUDA_DEVICE)
        self.wav2lip = Wav2LipPredictor(device=self.device)
        self.tts = TextToSpeech()
        self.face_enhancer = FaceExpressionEnhancer(device=self.device)

    @monitor_timing(process_type="full_pipeline")
    async def process_video(self, input_video_path: str, text: str) -> str:
        try:
            # Validate input video
            validate_video(input_video_path)
            
            # Generate speech from text
            audio_path = await self.tts.generate_speech(text)
            
            # Process video with Wav2Lip
            synced_video = await self.wav2lip.predict(input_video_path, audio_path)
            
            # Enhance facial expressions
            enhanced_video = await self.face_enhancer.enhance(synced_video)
            
            return enhanced_video
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

@router.post("/lipsync")
async def create_lipsync(
    video: UploadFile = File(...),
    text: str = Form(...)
):
    try:
        logger.info(f"Processing request for video: {video.filename}")
        REQUESTS_TOTAL.labels(endpoint="/lipsync", status="started").inc()
        
        # Save uploaded video to temporary file
        temp_video_path = Path(tempfile.mktemp(suffix='.mp4'))
        with open(temp_video_path, "wb") as buffer:
            buffer.write(await video.read())
        
        # Process video
        pipeline = LipSyncPipeline()
        result_path = await pipeline.process_video(str(temp_video_path), text)
        
        REQUESTS_TOTAL.labels(endpoint="/lipsync", status="success").inc()
        return FileResponse(result_path)
        
    except Exception as e:
        REQUESTS_TOTAL.labels(endpoint="/lipsync", status="error").inc()
        ERROR_COUNTER.labels(error_type="api_request").inc()
        logger.error(f"Request processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup temporary files
        if temp_video_path.exists():
            temp_video_path.unlink()