import requests
import configparser
import os

# Load config
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), '../../config/config.ini')
config.read(config_path)

# Get API endpoint and key from config
url = config['API']['SARVAM_API_ENDPOINT'] + "/text-to-speech"
api_key = config['API']['SARVAM_API_KEY']

def text_to_speech(text, output_file="output.wav"):
    
    payload = {
        "inputs": [text],
        "target_language_code": "hi-IN", 
        "speaker": "meera",
        "pitch": 0,
        "pace": 1.65,
        "loudness": 1.5,
        "speech_sample_rate": 8000,
        "enable_preprocessing": True,
        "model": "bulbul:v1"
    }
    
    headers = {
        "Content-Type": "application/json",
        'API-Subscription-Key': api_key
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    if response.status_code == 200:
        # Get base64 encoded audio from response
        audio_base64 = response.json()['audios'][0]
        
        # Decode base64 to binary
        import base64
        audio_binary = base64.b64decode(audio_base64)
        
        # Write to WAV file
        with open(output_file, "wb") as f:
            f.write(audio_binary)
        print(f"Audio saved to {output_file}")
        return True
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return False
