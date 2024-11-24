import subprocess
import os
import time
import signal
import atexit
from typing import Optional, Dict
from threading import Thread

class Speaker:
    """
    A class to manage virtual audio devices and stream audio from video files.
    """
    
    def __init__(self, device_name: str = "Virtual_Speaker"):
        """
        Initialize the Speaker class.
        
        Args:
            device_name (str): Name for the virtual audio device
        """
        self.device_name = device_name
        self.current_process: Optional[subprocess.Popen] = None
        self.active_streams: Dict[str, subprocess.Popen] = {}
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Initialize audio system
        self._setup_audio_system()
    
    def _setup_audio_system(self) -> None:
        """Set up the virtual audio system."""
        try:
            # Check ffmpeg installation
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            
            # Start PulseAudio if needed
            try:
                subprocess.run(['pulseaudio', '--check'], check=True)
            except subprocess.CalledProcessError:
                subprocess.run(['pulseaudio', '--start'], check=True)
                time.sleep(1)
            
            # Create virtual sink
            subprocess.run([
                'pactl', 'load-module', 'module-null-sink',
                f'sink_name={self.device_name}',
                f'sink_properties=device.description="{self.device_name}"'
            ], check=True)
            
            # Create virtual source
            subprocess.run([
                'pactl', 'load-module', 'module-virtual-source',
                'source_name=virtual_mic',
                f'master={self.device_name}.monitor',
                'source_properties=device.description="Virtual_Microphone"'
            ], check=True)
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to setup audio system: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during setup: {str(e)}")
    
    def play(self, video_path: str, block: bool = False) -> str:
        """
        Play audio from a video file.
        
        Args:
            video_path (str): Path to the video file
            block (bool): If True, blocks until playback completes
            
        Returns:
            str: Stream ID that can be used to stop the playback
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Generate unique stream ID
        stream_id = str(hash(video_path + str(time.time())))
        
        try:
            # Start FFmpeg process
            process = subprocess.Popen([
                'ffmpeg',
                '-i', video_path,
                '-f', 'pulse',
                self.device_name
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.active_streams[stream_id] = process
            
            if block:
                process.wait()
                self._cleanup_stream(stream_id)
            else:
                # Start monitoring thread
                Thread(target=self._monitor_stream, args=(stream_id,), daemon=True).start()
            
            return stream_id
            
        except Exception as e:
            if stream_id in self.active_streams:
                self._cleanup_stream(stream_id)
            raise RuntimeError(f"Failed to play audio: {str(e)}")
    
    def stop(self, stream_id: Optional[str] = None) -> None:
        """
        Stop audio playback.
        
        Args:
            stream_id (Optional[str]): Specific stream to stop, or None to stop all
        """
        if stream_id is None:
            # Stop all streams
            for sid in list(self.active_streams.keys()):
                self._cleanup_stream(sid)
        elif stream_id in self.active_streams:
            self._cleanup_stream(stream_id)
    
    def _monitor_stream(self, stream_id: str) -> None:
        """Monitor a stream and clean up when it finishes."""
        if stream_id in self.active_streams:
            self.active_streams[stream_id].wait()
            self._cleanup_stream(stream_id)
    
    def _cleanup_stream(self, stream_id: str) -> None:
        """Clean up a specific stream."""
        if stream_id in self.active_streams:
            process = self.active_streams[stream_id]
            try:
                process.terminate()
                process.wait(timeout=1)
            except (subprocess.TimeoutExpired, Exception):
                process.kill()
            finally:
                del self.active_streams[stream_id]
    
    def cleanup(self) -> None:
        """Clean up all resources."""
        # Stop all active streams
        self.stop()
        
        try:
            # Get list of loaded modules
            loaded_modules = subprocess.check_output(['pactl', 'list', 'modules']).decode()
            
            # Find and unload relevant modules
            for line in loaded_modules.split('\n'):
                if self.device_name in line or "virtual_mic" in line:
                    if 'Module #' in line:
                        module_num = line.split('#')[1].strip()
                        subprocess.run(['pactl', 'unload-module', module_num], check=True)
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle system signals."""
        self.cleanup()
        exit(0)
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
