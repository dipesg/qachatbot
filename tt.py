import soundfile as sf
import numpy as np
import requests

SERVER_HOST= "192.168.1.70"
SERVER_PORT = 8000
API_PATH = "/generate"                                                       
#192.168.1.70
url = SERVER_HOST+":"+str(SERVER_PORT)+API_PATH
resp = requests.post(url, data='नमस्ते'.encode('utf-8'))                                    
data = resp.json()                                                                          
sf.write('tts_audio.flac', data, 16000)