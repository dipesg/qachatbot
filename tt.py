import soundfile as sf
import numpy as np
import requests
from translate import Translator
SERVER_HOST= "http://192.168.88.253"
SERVER_PORT = 8000

def call_asr(audio, api_path = '/inference'):
    #print(audio)
    url = SERVER_HOST+":"+str(SERVER_PORT)+api_path
    files = {'file' : open(audio,'rb')}
    resp = requests.post(url,files=files )                                    
    data = resp.json()['text']
    return data                                                                     

def call_tts_nepali(text, api_path='/generate'):
    url = SERVER_HOST+":"+str(SERVER_PORT)+api_path
    resp = requests.post(url, data=text.encode('utf-8'))                                    
    data = np.array(resp.json())
    # print(type(data))
    sf.write('/mnt/media/wiseyak/qachatbot/qachatbot/response.wav', data, 16000)
    return True
    # pass

def trans(text):
    translator = Translator(to_lang='ne')
    translation = translator.translate(text)
    print("Translation",translation)
    return translation


# call_tts_nepali("राष्ट्रपति रामचन्द्र पौडेलले आज (शुक्रबार) बेलुकी ६ बजे संसदमा नीति")
trans("Over the last couple of decades, the technological advances in storage and processing power have enabled some innovative products based on machine learning")