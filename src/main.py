import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from audio import AudioCapture
from transcribe import MLXTranscriber
from config import *

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global components
transcription_results = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global service
    try:
        await service.initialize()
        logger.info("API server started successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        sys.exit(1)
    finally:
        # Shutdown
        await service.cleanup()
        logger.info("API server shut down")

# FastAPI app
app = FastAPI(
    title="MLX Whisper Real-time Transcription API",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranscriptionService:
    def __init__(self):
        self.audio_capture = AudioCapture()
        self.transcriber = MLXTranscriber()
        self.is_running = False
        self.transcription_task = None

    async def initialize(self):
        """Initialize the transcription service"""
        try:
            logger.info("Initializing transcription service...")
            await self.transcriber.load_model()
            self.audio_capture.get_device_info()
            logger.info("Transcription service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize transcription service: {e}")
            raise

    async def start_transcription(self):
        """Start real-time transcription"""
        if self.is_running:
            logger.warning("Transcription already running")
            return

        self.is_running = True
        logger.info("Starting real-time transcription")
        
        try:
            async for audio_chunk in self.audio_capture.start_recording():
                if not self.is_running:
                    break
                
                # Transcribe audio chunk
                result = await self.transcriber.transcribe(audio_chunk)
                
                if result["text"].strip():
                    logger.info(f"Transcription: {result['text']}")
                    transcription_results.append(result)
                    
                    # Keep only last 50 results to prevent memory bloat
                    if len(transcription_results) > 50:
                        transcription_results.pop(0)
                        
        except Exception as e:
            logger.error(f"Transcription error: {e}")
        finally:
            self.is_running = False
            logger.info("Transcription stopped")

    async def stop_transcription(self):
        """Stop real-time transcription"""
        self.is_running = False
        self.audio_capture.stop_recording()
        logger.info("Stopping transcription service")

    async def cleanup(self):
        """Cleanup resources"""
        await self.stop_transcription()
        self.transcriber.unload_model()
        logger.info("Transcription service cleaned up")

# Global service instance
service = TranscriptionService()

@app.get("/")
async def root():
    return {"message": "MLX Whisper Real-time Transcription API"}

@app.post("/start")
async def start_transcription():
    """Start real-time transcription"""
    global service
    
    if service.is_running:
        return {"status": "already_running"}
    
    # Start transcription in background task
    service.transcription_task = asyncio.create_task(service.start_transcription())
    return {"status": "started"}

@app.post("/stop")
async def stop_transcription():
    """Stop real-time transcription"""
    global service
    
    await service.stop_transcription()
    if service.transcription_task:
        service.transcription_task.cancel()
        try:
            await service.transcription_task
        except asyncio.CancelledError:
            pass
    
    return {"status": "stopped"}

@app.get("/results")
async def get_transcription_results():
    """Get recent transcription results"""
    return {
        "results": transcription_results,
        "count": len(transcription_results),
        "is_running": service.is_running
    }

@app.delete("/results")
async def clear_transcription_results():
    """Clear transcription results"""
    global transcription_results
    transcription_results.clear()
    return {"status": "cleared"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": service.transcriber.is_loaded,
        "is_transcribing": service.is_running,
        "audio_devices": len(service.audio_capture.get_device_info() or [])
    }

@app.get("/status")
async def get_status():
    """Get detailed service status"""
    devices = service.audio_capture.get_device_info()
    return {
        "transcription": {
            "is_running": service.is_running,
            "model_name": service.transcriber.model_name,
            "language": service.transcriber.language,
            "diarization_enabled": ENABLE_DIARIZATION,
            "diarization_loaded": service.transcriber.diarization.is_loaded if service.transcriber.diarization else False
        },
        "audio": {
            "sample_rate": service.audio_capture.sample_rate,
            "chunk_size": service.audio_capture.chunk_size,
            "overlap_size": service.audio_capture.overlap_size,
            "devices_available": len(devices) if devices else 0
        },
        "results_count": len(transcription_results)
    }

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

async def main():
    """Main entry point"""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting MLX Whisper Real-time Transcription Service")
    
    # Run FastAPI server
    config = uvicorn.Config(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level=LOG_LEVEL.lower()
    )
    server = uvicorn.Server(config)
    
    try:
        await server.serve()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        await service.cleanup()

if __name__ == "__main__":
    asyncio.run(main())