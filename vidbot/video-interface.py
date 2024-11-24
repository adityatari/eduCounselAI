import cv2
import numpy as np
import time
from speaker_ffmpeg import Speaker


class Videoplayer:
    def __init__(self):
        self.current_video = None
        
    def play(self, video_path=None):
        self.current_video = video_path    

    def play_video_or_static(self, static_image_path="extras/potrait_shot_man_3.png"):
        # If no video and no static image provided, create a blank frame
        try:
            while True:
                if self.current_video is None and static_image_path is None:
                    static_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                # If static image provided, use it
                elif static_image_path is not None:
                    static_frame = cv2.imread(static_image_path)
                    while self.current_video is None:
                        cv2.imshow('Video Player', static_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                
                # If video path provided, play video
                if self.current_video is not None:
                    cap = cv2.VideoCapture(self.current_video)
                    speaker = Speaker()
                    stream_id = speaker.play(self.current_video)
                    if not cap.isOpened():
                        print(f"Error: Could not open video file {self.current_video}")
                        return
                    
                    # Get video properties
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_time = 1/fps if fps > 0 else 1/30  # Default to 30fps if fps not available
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            # End video playback instead of looping
                            break
                        
                        cv2.imshow('Video Player', frame)
                        
                        # Break loop if 'q' is pressed
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                            
                        # Control frame rate
                        time.sleep(frame_time)
                    
                    cap.release()
                    speaker.stop(stream_id)
       
        except KeyboardInterrupt:
            print("\nStopping voice interface...")
            # Stop all speaker
            speaker.stop()
            cv2.destroyAllWindows()
