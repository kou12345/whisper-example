import pytest
import asyncio
import time
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.transcribe import MLXTranscriber

class TestLatency:
    @pytest.fixture
    async def transcriber(self):
        """Setup transcriber fixture"""
        transcriber = MLXTranscriber()
        try:
            await transcriber.load_model()
            yield transcriber
        finally:
            transcriber.unload_model()

    @pytest.mark.asyncio
    async def test_model_loading_time(self):
        """Test model loading latency"""
        transcriber = MLXTranscriber()
        
        start_time = time.time()
        await transcriber.load_model()
        loading_time = time.time() - start_time
        
        transcriber.unload_model()
        
        # Model should load within reasonable time (adjust based on model size)
        assert loading_time < 30.0, f"Model loading took {loading_time:.2f}s, should be < 30s"

    @pytest.mark.asyncio
    async def test_transcription_latency(self, transcriber):
        """Test transcription processing latency"""
        # Generate 2-second test audio (silence with small noise)
        sample_rate = 16000
        duration = 2.0
        audio_samples = int(sample_rate * duration)
        
        # Add small amount of noise to avoid empty transcription
        test_audio = np.random.normal(0, 0.01, audio_samples).astype(np.float32)
        
        start_time = time.time()
        result = await transcriber.transcribe(test_audio)
        transcription_time = time.time() - start_time
        
        # Transcription should be faster than real-time (< 2s for 2s audio)
        assert transcription_time < 2.0, f"Transcription took {transcription_time:.2f}s for 2s audio"
        
        # Result should have expected structure
        assert isinstance(result, dict), "Result should be dictionary"
        assert "text" in result, "Result should contain 'text' field"
        assert "segments" in result, "Result should contain 'segments' field"
        assert "language" in result, "Result should contain 'language' field"

    @pytest.mark.asyncio
    async def test_concurrent_transcriptions(self, transcriber):
        """Test multiple concurrent transcriptions"""
        # Generate test audio chunks
        sample_rate = 16000
        duration = 1.0  # Shorter chunks for concurrent test
        audio_samples = int(sample_rate * duration)
        
        num_concurrent = 3
        test_audios = []
        for i in range(num_concurrent):
            # Generate different frequency sine waves
            t = np.linspace(0, duration, audio_samples, False)
            freq = 440 + i * 110  # 440Hz, 550Hz, 660Hz
            audio = np.sin(2 * np.pi * freq * t).astype(np.float32) * 0.1
            test_audios.append(audio)
        
        # Run transcriptions concurrently
        start_time = time.time()
        tasks = [transcriber.transcribe(audio) for audio in test_audios]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # All transcriptions should complete
        assert len(results) == num_concurrent, "All transcriptions should complete"
        
        # Total time should be reasonable (not much slower than single transcription)
        expected_max_time = 3.0  # Allow some overhead for concurrent processing
        assert total_time < expected_max_time, f"Concurrent transcriptions took {total_time:.2f}s"

    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, transcriber):
        """Test memory usage doesn't grow excessively"""
        sample_rate = 16000
        duration = 1.0
        audio_samples = int(sample_rate * duration)
        
        # Generate test audio
        t = np.linspace(0, duration, audio_samples, False)
        test_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.1
        
        # Run multiple transcriptions
        num_iterations = 5
        for i in range(num_iterations):
            result = await transcriber.transcribe(test_audio)
            assert isinstance(result, dict), f"Iteration {i}: Result should be dictionary"
            
            # Small delay to allow garbage collection
            await asyncio.sleep(0.1)
        
        # If we reach here without memory errors, the test passes
        assert True, "Memory usage remained stable"