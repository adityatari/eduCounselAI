import sounddevice as sd
import numpy as np
import webrtcvad
import wave
import threading
import queue
import time
from collections import deque
import requests
import subprocess
import json
import os
from scipy import signal

class BrowserAudioProcessor:
    def __init__(self, sample_rate=16000, save_dir='recordings', use_vad=True):
        self.sample_rate = sample_rate
        self.save_dir = save_dir
        self.use_vad = use_vad  # New parameter to control VAD
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Only initialize VAD if we're using it
        self.vad = webrtcvad.Vad(3) if use_vad else None
        self.frame_duration = 30  # ms
        self.buffer = deque(maxlen=500)
        self.is_recording = True
        self.audio_queue = queue.Queue()
        self.chunk_duration = 10  # seconds
        self.silence_threshold = 0.8
        
        # Enhanced VAD parameters
        self.energy_threshold = 0.000001  # Adjustable energy threshold
        self.speech_frames_threshold = 8  # Minimum frames to consider as speech
        self.max_silence_frames = 20  # Maximum silence frames within speech
        self.pre_speech_frames = 5  # Number of frames to keep before speech
        self.speech_buffer = []
        self.temp_buffer = deque(maxlen=self.pre_speech_frames)
        self.silence_counter = 0
        self.speech_frame_counter = 0
        self.is_speech_active = False

    def calculate_energy(self, audio_frame):
        """Calculate the energy of an audio frame."""
        return np.mean(np.abs(audio_frame))

    def is_speech_frame(self, audio_frame):
        """
        Enhanced speech detection using both WebRTC VAD and energy levels.
        Returns True if the frame contains speech.
        """
        # Convert to the required format for WebRTC VAD
        audio_frame_int16 = (audio_frame * 32767).astype(np.int16)
        
        # Calculate frame energy
        frame_energy = self.calculate_energy(audio_frame)
        
        # Get VAD decision
        try:
            vad_decision = self.vad.is_speech(audio_frame_int16.tobytes(), self.sample_rate)
        except Exception:
            vad_decision = False
            
        # Combine VAD and energy threshold
        return vad_decision and frame_energy > self.energy_threshold

    def process_frame_for_speech(self, audio_frame):
        """
        Process a single frame to detect speech and manage speech segments.
        Returns True if the frame should be included in the current speech segment.
        """
        is_speech = self.is_speech_frame(audio_frame)
        
        if is_speech:
            self.speech_frame_counter += 1
            self.silence_counter = 0
            
            if not self.is_speech_active and self.speech_frame_counter >= self.speech_frames_threshold:
                self.is_speech_active = True
                # Include pre-speech frames
                return True, True  # (is_speech_segment, include_previous_frames)
                
        else:
            self.silence_counter += 1
            if self.is_speech_active:
                if self.silence_counter >= self.max_silence_frames:
                    self.is_speech_active = False
                    self.speech_frame_counter = 0
                    return False, False
            else:
                self.speech_frame_counter = max(0, self.speech_frame_counter - 1)
                
        return self.is_speech_active, False

    def _process_audio(self):
        """Process audio stream and detect speech segments."""
        frame_size = int(self.sample_rate * self.frame_duration / 1000)
        current_segment = []
        
        while self.is_recording:
            if len(self.buffer) >= frame_size:
                # Get frame from buffer
                frame = np.array(list(self.buffer)[:frame_size])
                self.buffer.rotate(-frame_size)
                
                if self.use_vad:
                    # Original VAD logic
                    self.temp_buffer.append(frame)
                    is_speech_segment, include_previous = self.process_frame_for_speech(frame)
                    
                    if is_speech_segment:
                        if include_previous:
                            valid_frames = [f for f in list(self.temp_buffer)[:-1] if f.size > 0]
                            current_segment.extend(valid_frames)
                        current_segment.extend([frame])
                        
                        if len(current_segment) >= self.sample_rate * self.chunk_duration:
                            valid_frames = [f for f in current_segment if f.size > 0 and f.ndim == 1]
                            if valid_frames:
                                try:
                                    self._save_and_process_chunk(np.concatenate(valid_frames))
                                except ValueError as e:
                                    print(f"Error processing chunk: {e}")
                            current_segment = []
                            
                    elif len(current_segment) > 0:
                        if len(current_segment) > frame_size * 3:
                            valid_frames = [f for f in current_segment if f.size > 0 and f.ndim == 1]
                            if valid_frames:
                                try:
                                    self._save_and_process_chunk(np.concatenate(valid_frames))
                                except ValueError as e:
                                    print(f"Error processing chunk: {e}")
                        current_segment = []
                
                else:
                    # No VAD - continuously record in chunks
                    current_segment.append(frame)
                    if len(current_segment) >= self.sample_rate * self.chunk_duration:
                        valid_frames = [f for f in current_segment if f.size > 0 and f.ndim == 1]
                        if valid_frames:
                            try:
                                self._save_and_process_chunk(np.concatenate(valid_frames))
                            except ValueError as e:
                                print(f"Error processing chunk: {e}")
                        current_segment = []
                    
            time.sleep(0.01)

    def _save_and_process_chunk(self, audio_chunk):
        """Save and process a speech chunk."""
        # Apply noise reduction
        filtered_chunk = self._reduce_noise(audio_chunk)
        
        # Normalize audio
        normalized_chunk = np.int16(filtered_chunk * 32767)
        
        # Save to WAV file in recordings directory
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.save_dir, f"speech_chunk_{timestamp}.wav")
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(normalized_chunk.tobytes())
        
        # Add to processing queue
        self.audio_queue.put(filename)
        threading.Thread(target=self._process_queue).start()

    def _reduce_noise(self, audio_chunk):
        """Simple noise reduction using a high-pass filter."""
        nyquist = self.sample_rate // 2
        cutoff = 100  # Hz
        normalized_cutoff = cutoff / nyquist
        b, a = signal.butter(4, normalized_cutoff, btype='high', analog=False)
        return signal.filtfilt(b, a, audio_chunk)

    def start_recording(self):
        self.is_recording = True
        threading.Thread(target=self._record_audio).start()
        threading.Thread(target=self._process_audio).start()
        return True

    def _record_audio(self):
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Error: {status}")
            audio_data = np.mean(indata, axis=1) if len(indata.shape) > 1 else indata
            self.buffer.extend(audio_data)
            
        try:
            with sd.InputStream(
                callback=audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                dtype=np.float32
            ):
                while self.is_recording:
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"Error recording audio: {str(e)}")
            self.is_recording = False

    def stop_recording(self):
        self.is_recording = False

    def _process_queue(self):
        while not self.audio_queue.empty():
            filename = self.audio_queue.get()
            self._send_to_speech_to_text_api(filename)
            # Remove the os.remove(filename) line to keep the recordings

    def _send_to_speech_to_text_api(self, filename):
        """
        Send audio file to speech-to-text API
        Replace this with your preferred API (e.g., Google Cloud, AWS Transcribe, Whisper API)
        """
        try:
            # Example using a generic REST API
            with open(filename, 'rb') as audio_file:
                files = {'file': audio_file}
                response = requests.post(
                    'YOUR_SPEECH_TO_TEXT_API_ENDPOINT',
                    files=files,
                    headers={'Authorization': 'YOUR_API_KEY'}
                )
                
                if response.status_code == 200:
                    transcription = response.json()
                    print(f"Transcription: {transcription['text']}")
                else:
                    print(f"Error in API request: {response.status_code}")
                    
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")

if __name__ == "__main__":
    # Check for PulseAudio access instead of requiring root
    try:
        # Test PulseAudio access by listing sinks
        subprocess.run(["pactl", "list", "sinks"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("Error: Unable to access PulseAudio. Please ensure you have the correct permissions.")
        print("You may need to add your user to the 'pulse' and 'pulse-access' groups:")
        print("sudo usermod -aG pulse,pulse-access $USER")
        exit(1)
        
    # Initialize with VAD disabled - will record continuously
    processor = BrowserAudioProcessor(save_dir='my_recordings', use_vad=False)
    
    try:
        if processor.start_recording():
            print("Recording... Press Ctrl+C to stop")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        
        processor.stop_recording()
        print("\nRecording stopped")