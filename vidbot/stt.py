import requests

import configparser
import os

config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), '../../config/config.ini')
config.read(config_path)
api_key = config['ASSEMBLYAI']['API_KEY']


def get_transcript(audio_path):
    """
    Convert audio to transcript using AssemblyAI
    
    Args:
        audio_path (str): Path to audio file
        api_key (str): AssemblyAI API key
    
    Returns:
        str: Transcribed text
    """
    # Upload
    upload_url = 'https://api.assemblyai.com/v2/upload'
    headers = {'Authorization': api_key}
    
    with open(audio_path, 'rb') as file:
        response = requests.post(upload_url, headers=headers, data=file)
    
    if response.status_code != 200:
        raise Exception(f"Upload error: {response.status_code} - {response.text}")
    
    audio_url = response.json()['upload_url']
    
    # Transcribe
    transcript_url = "https://api.assemblyai.com/v2/transcript"
    headers = {
        "authorization": api_key,
        "content-type": "application/json"
    }
    data = {"audio_url": audio_url}

    response = requests.post(transcript_url, json=data, headers=headers)
    transcript_id = response.json()['id']
    
    # Poll for results
    polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    while True:
        transcription_result = requests.get(polling_endpoint, headers=headers).json()
        
        if transcription_result['status'] == "completed":
            return transcription_result['text']
        elif transcription_result['status'] == "error":
            raise Exception(f"Sorry we could not undertand what you said having an issue with: {transcription_result['error']}")
    