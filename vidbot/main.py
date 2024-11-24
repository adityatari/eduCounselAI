import os
import queue
import threading
from dataclasses import dataclass
import cv2
import time
import numpy as np
from typing import Optional
import subprocess
from utils.helper_functions import process_text,search_programs_from_sqlite
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import configparser
import requests
from tts import text_to_speech as tts


# Load config
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), '../../config/config.ini')
config.read(config_path)

# Get API endpoint and key from config
url = config['API']['SARVAM_API_ENDPOINT'] + "/text-to-speech"
api_key = config['API']['SARVAM_API_KEY']

@dataclass
class BotState:
    current_video: Optional[np.ndarray] = None
    is_listening: bool = False
    is_processing: bool = False
    should_exit: bool = False




class VideoBot:
    def __init__(self):
        self.state = BotState()
        self.audio_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.video_queue = queue.Queue()
        self.source_image_path = "./extras/potrait_shot_man_3.png"
        
        # Initialize components
        self.init_models()
        
    def init_models(self):
        """Initialize all required models"""
        model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Adjust for 1B if you have it
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

        pass

    def start(self):
        """Start all processing threads"""
        threads = [
            threading.Thread(target=self.audio_listener_thread),
            threading.Thread(target=self.processing_thread),
            threading.Thread(target=self.video_player_thread),
        ]
        
        for thread in threads:
            thread.daemon = True
            thread.start()
            
        return threads

    def audio_listener_thread(self):
        """Continuously listen for audio input"""
        while not self.state.should_exit:
            if not self.state.is_listening:
                # TODO: Implement audio recording
                audio_data = self.record_audio()
                self.audio_queue.put(audio_data)
                self.state.is_listening = True

    def processing_thread(self):
        """Handle the main processing pipeline"""
        chat_history = ""
        while not self.state.should_exit:
            if not self.audio_queue.empty():
                self.state.is_processing = True
                
                # Get audio and process
                text = self.audio_queue.get()
                
                # Speech to text
                # text = self.speech_to_text(audio_data)
                
                # Get LLM response
                response = self.get_llm_response(chat_history, text)
                chat_history += f"User: {text}\nChatbot: {response}\n"
                text_list = process_text(response, max_chars=100)
                for idx, text in enumerate(text_list):
                    audio_path = self.text_to_speech(text, f"./vidbot/llm_speech_recordings/ai_response_{idx}.wav")
                    # Generate video
                    video = self.generate_video(audio_path, self.source_image_path , result_dir="./video_outputs/")
                
                    # Queue the video
                    self.video_queue.put(video)
                
                self.state.is_processing = False
                self.state.is_listening = False

    def video_player_thread(self):
        """Display the video frames"""
        while not self.state.should_exit:
            if not self.video_queue.empty():
                # Get new video
                self.state.current_video = self.video_queue.get()
            
            if self.state.current_video is not None:
                # Play current frame
                # TODO: Implement proper frame timing
                cv2.imshow('Video Bot', self.state.current_video)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.state.should_exit = True


    def get_llm_response(self, chat_history, user_input):
        """Get response from LLM"""
        # TODO: Implement LLM interaction
        prompt = (
            "You are a helpful and knowledgeable guidance counselor for students.\n"
            f"Here is the conversation so far:\n{chat_history}\n"
            f"User: {user_input}\n"
            "Provide a helpful and detailed response:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)
        outputs = self.model.generate(**inputs, max_length=1024, temperature=0.7, top_p=0.9)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # chat_history += f"User: {user_input}\nChatbot: {response}\n"
        return response.split("Provide a helpful and detailed response:")[-1].strip()


    def text_to_speech(self, text, output_file="output.wav"):
        return tts(text, output_file)

    def generate_video(self,audio_path, source_image_path, result_dir="./video_outputs/"):
        """Generate lip-synced video"""
        # TODO: Implement video generation
        # pass
        try:
            os.makedirs(result_dir, exist_ok=True)
            
            # Construct the command
            command = [
                "python", "SadTalker/inference.py",
                "--driven_audio", audio_path,
                "--source_image", source_image_path,
                "--result_dir", result_dir,
                "--still"
            ]
            process = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error generating video: {e}")
            print(f"Error output: {e.stderr}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None                    
     

    def record_audio(self):
        """Record audio input"""
        # TODO: Implement audio recording
        pass

def main():
    bot = VideoBot()
    threads = bot.start()
    
    try:
        # Keep main thread alive
        while not bot.state.should_exit:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutting down...")
        bot.state.should_exit = True
        
    # Wait for threads to finish
    for thread in threads:
        thread.join()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
