from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ChatVectorDBChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.llm import LLMChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import FAISS
import gradio as gr
from gtts import gTTS
from base64 import b64encode
from io import BytesIO
import time
import pyttsx3
from dotenv import load_dotenv
import gradio as gr 
from langchain.chat_models import ChatOpenAI
import pyttsx3
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyMuPDFLoader
import textwrap
import re
import aiohttp
import asyncio
from gtts import gTTS
import speech_recognition as sr
from speech_recognition import Recognizer
import requests
from langchain.chains import ConversationalRetrievalChain
r = sr.Recognizer()

class Demo:
    def __init__(self):
        self.chat_history = []
        self.prev_bot_response = ''
        self.question = ''
        self.persist_directory1 = './pdf_db'
        self.embedding = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name='gpt-3.5-turbo-0301', temperature=0, streaming=True, verbose=True)
        self.streaming_llm = ChatOpenAI(model_name='gpt-3.5-turbo-0301', streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=0)
        loader = PyMuPDFLoader("docs/report1.pdf")
        self.data = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(self.data)
        db = Chroma.from_documents(docs, self.embedding, persist_directory=self.persist_directory1)
        self.retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
        
    def chatbot(self, query):
        # create a chain to answer questions 
        qa = ConversationalRetrievalChain.from_llm(self.llm, self.retriever)
        chat_history = []
        result = qa({"question": query, "chat_history": chat_history})
        print(result["answer"])
        return result["answer"]
    
    def stt(self) -> str:
        """Converts speech to text.
        Args:
            audio: record of user speech
        Returns:
            text (str): recognized speech of user
        """
        # Create a Recognizer object
        r = Recognizer()
        # Open the audio file
        with sr.Microphone() as source:
            # Listen for the data (load audio to memory)
            audio_data = r.record(source)
            # Transcribe the audio using Google's speech-to-text API
            text = r.recognize_google(audio_data, language="en-US")
            print("You said: ", text)
            
        return text

    def recognize_speech_from_file(self, file_path):
        # Create a recognizer instance
        recognizer = sr.Recognizer()

        # Use the audio file as the audio source
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)

        # Use Google Speech Recognition to transcribe the audio
        try:
            text = recognizer.recognize_google(audio, language="en-US")
            print("Transcription: ", text)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

        return text

# Initialize Chatbot class object
demo = Demo()

def split_streaming_response(text):
    response = demo.chatbot(text)
    # Initialize variables
    buffer = ""
    result = []

    # Define a function to split the buffer into groups of 10 words
    def split_buffer():
        nonlocal buffer, result
        words = re.findall(r'\w+', buffer)
        groups_of_10 = [words[i:i+10] for i in range(0, len(words), 10)]
        result += [' '.join(group) for group in groups_of_10]
        buffer = ""

    # Iterate over each character in the response
    for char in response:
        buffer += char
        if char == " ":
            # Split the buffer when a space is encountered
            if len(re.findall(r'\w+', buffer)) >= 10:
                split_buffer()

    # Split any remaining words in the buffer
    if len(re.findall(r'\w+', buffer)) > 0:
        split_buffer()
        
    print("Result :", result)
    return result
r = sr.Recognizer()
SERVER_HOST= "192.168.1.70"
SERVER_PORT = 8000
API_PATH = "/generate" 

async def convert_text_to_speech(sentence, session):
    # Make the API call to convert the sentence to speech
    url = "http://"+SERVER_HOST+":"+str(SERVER_PORT)+API_PATH
    resp = requests.post(url, data='नमस्ते'.encode('utf-8'))
    async with session.get('https://your_tts_api.com', params={'text': sentence}) as response:
        audio = await response.read()

    return audio
async def call_gtts_api(text, index):
    loop = asyncio.get_running_loop()
    tts = gTTS(text=text, lang='en')
    tts.save(f'output_{index}.mp3')

    with open(f'output_{index}.mp3', 'rb') as f:
        audio_content = f.read()

    return audio_content

async def transcribe_audio(audio_content):
    with sr.AudioFile(audio_content) as source:
        audio = r.record(source)
    transcript = r.recognize_google(audio)
    return transcript

async def main(text):
    sentences = split_streaming_response(text)
    audio_tasks = []
    for i, sentence in enumerate(sentences, start=1):
        audio_task = asyncio.create_task(call_gtts_api(sentence, i))
        audio_tasks.append(audio_task)

    audio_contents = await asyncio.gather(*audio_tasks)

    transcript_tasks = []
    for audio_content in audio_contents:
        transcript_task = asyncio.create_task(transcribe_audio(audio_content))
        transcript_tasks.append(transcript_task)

    transcripts = await asyncio.gather(*transcript_tasks)
    return transcripts

# Define your Gradio interface components
input_text = gr.inputs.Textbox(label="Input Text")
input_aud = gr.inputs.Audio(source="microphone", type="filepath")
output_text = gr.outputs.Textbox(label="Output Text")
output_aud = gr.outputs.Audio(type="filepath")

# Define your Gradio interface function
async def process_text(audio=None, text=None):
    if audio:
        message = demo.recognize_speech_from_file(audio)
        audio_contents = await main(message)
        # Process the audio contents as needed
        # ...

        return "Processed output"
    else:
        message = text
        audio_contents = await main(message)
        # Process the audio contents as needed
        # ...

        return "Processed output"

# Create your Gradio interface
gr.Interface(fn=process_text, inputs=[input_aud,input_text], outputs=output_text).launch()