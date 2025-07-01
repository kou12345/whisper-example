#!/usr/bin/env python3
"""
Audio test script to check microphone access
"""
import sounddevice as sd
import numpy as np
import time

def test_audio_devices():
    """List available audio devices"""
    print("Available audio devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"  {i}: {device['name']} (inputs: {device['max_input_channels']})")

def test_microphone():
    """Test microphone recording"""
    print("\nTesting microphone recording...")
    
    try:
        # Record for 3 seconds
        duration = 3
        sample_rate = 16000
        
        print(f"Recording for {duration} seconds at {sample_rate}Hz...")
        audio_data = sd.rec(int(duration * sample_rate), 
                           samplerate=sample_rate, 
                           channels=1, 
                           dtype=np.float32)
        sd.wait()  # Wait until recording is finished
        
        # Check if we got audio data
        max_level = np.max(np.abs(audio_data))
        avg_level = np.mean(np.abs(audio_data))
        
        print(f"Recording complete!")
        print(f"  Data shape: {audio_data.shape}")
        print(f"  Max level: {max_level:.4f}")
        print(f"  Average level: {avg_level:.4f}")
        
        if max_level > 0.001:
            print("✅ Audio data detected - microphone is working!")
        else:
            print("❌ No audio data detected - check microphone permissions")
            
    except Exception as e:
        print(f"❌ Error testing microphone: {e}")

def test_callback_recording():
    """Test callback-based recording"""
    print("\nTesting callback-based recording...")
    
    audio_buffer = []
    
    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        audio_buffer.extend(indata.flatten())
        print(f"Callback: received {frames} frames, buffer size: {len(audio_buffer)}")
    
    try:
        with sd.RawInputStream(
            samplerate=16000,
            channels=1,
            dtype=np.float32,
            blocksize=1024,
            callback=audio_callback
        ) as stream:
            print("Recording with callback for 3 seconds...")
            time.sleep(3)
            
        print(f"Final buffer size: {len(audio_buffer)}")
        if len(audio_buffer) > 0:
            max_level = np.max(np.abs(audio_buffer))
            print(f"Max level: {max_level:.4f}")
            if max_level > 0.001:
                print("✅ Callback recording is working!")
            else:
                print("❌ No audio detected in callback")
        else:
            print("❌ No data in callback buffer")
            
    except Exception as e:
        print(f"❌ Error with callback recording: {e}")

if __name__ == "__main__":
    print("🎤 Audio System Test")
    print("=" * 50)
    
    test_audio_devices()
    test_microphone()
    test_callback_recording()
    
    print("\n" + "=" * 50)
    print("If you see '❌ No audio data detected', please:")
    print("1. Check macOS System Preferences → Security & Privacy → Privacy → Microphone")
    print("2. Make sure Terminal or your Python app has microphone access")
    print("3. Try speaking louder or closer to the microphone")