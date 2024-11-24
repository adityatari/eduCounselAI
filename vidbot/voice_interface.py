import pyaudio
import webrtcvad
import wave
from collections import deque
import time
import threading
import os
import configparser

class VoiceInterface:
    @staticmethod
    def list_audio_devices():
        """List all available audio input devices"""
        audio = pyaudio.PyAudio()
        info = []
        
        for i in range(audio.get_device_count()):
            device_info = audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:  # Only input devices
                info.append({
                    'index': i,
                    'name': device_info['name'],
                    'channels': device_info['maxInputChannels'],
                    'sample_rate': int(device_info['defaultSampleRate'])
                })
        
        audio.terminate()
        return info

    def __init__(self, sample_rate=16000, frame_duration=30, device_index=None):
        # Load config
        self.config = configparser.ConfigParser()
        self.config.read('../../config/config.ini')
        
        # Audio configurations
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration  # in milliseconds
        self.frame_size = int(sample_rate * frame_duration / 1000)  # samples per frame
        
        # VAD setup
        self.vad = webrtcvad.Vad(3)  # aggressiveness level 3 (0-3)
        
        # Audio recording setup
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Buffer for storing audio data
        self.audio_buffer = deque(maxlen=int((sample_rate * 60) / self.frame_size))  # max 60 seconds
        
        # State variables
        self.is_recording = False
        self.speech_detected = False
        self.silence_start = None
        self.SILENCE_THRESHOLD = 1.0  # seconds
        
        # Add output directory configuration
        self.output_dir = "recordings"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Add new audio tracking
        self.new_audio_available = False
        self.latest_transcript = None
        
        self.device_index = device_index
        
    def start_recording(self):
        """Start recording audio from specified or default microphone"""
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.frame_size,
            input_device_index=self.device_index
        )
        
        self.is_recording = True
        threading.Thread(target=self._record_audio).start()
        
    def _record_audio(self):
        """Continuously record and process audio"""
        while self.is_recording:
            try:
                audio_chunk = self.stream.read(self.frame_size)
                is_speech = self.vad.is_speech(audio_chunk, self.sample_rate)
                
                if is_speech:
                    self.speech_detected = True
                    self.silence_start = None
                    self.audio_buffer.append(audio_chunk)
                elif self.speech_detected:
                    if self.silence_start is None:
                        self.silence_start = time.time()
                    elif time.time() - self.silence_start > self.SILENCE_THRESHOLD:
                        self._process_speech()
                        self.speech_detected = False
                        self.audio_buffer.clear()
            except Exception as e:
                print(f"Error recording audio: {e}")
                break
                
    def _process_speech(self):
        """Process the recorded speech and save as WAV file"""
        if not self.audio_buffer:
            return
            
        # Combine all audio chunks
        audio_data = b''.join(list(self.audio_buffer))
        
        # Generate timestamp for unique filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        wav_filename = f"{self.output_dir}/speech_{timestamp}.wav"
        
        # Save audio to WAV file
        with wave.open(wav_filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data)
            
        print(f"Saved audio to: {wav_filename}")
        
        # Send to STT API if needed
        self._send_to_stt_api(wav_filename)
        
    def has_new_audio(self):
        """Check if new audio transcript is available"""
        return self.new_audio_available
        
    def get_audio(self):
        """Get the latest transcript and reset the flag"""
        if self.new_audio_available:
            self.new_audio_available = False
            transcript = self.latest_transcript
            self.latest_transcript = None
            return transcript
        return None
        
    def _send_to_stt_api(self, audio_file):
        """Send audio file to Sarvam STT API"""
        try:
            import stt
            transcript = stt.get_transcript(audio_file)
            self.latest_transcript = transcript
            self.new_audio_available = True
            return transcript
                
        except Exception as e:
            print(f"Error in speech to text API call: {e}")
            return None
        
    def stop_recording(self):
        """Stop recording and clean up"""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        

def main():
    voice_interface = VoiceInterface()
    print("Starting voice interface... Press Ctrl+C to stop")
    
    try:
        voice_interface.start_recording()
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping voice interface...")
        voice_interface.stop_recording()

if __name__ == "__main__":
    main()
