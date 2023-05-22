from langchain.evaluation.loading import load_dataset
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from langchain.chains import SQLDatabaseSequentialChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.llms import OpenAIChat
from langchain.chains import ChatVectorDBChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
import json
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import FAISS
import gradio as gr
import speech_recognition as sr
from speech_recognition import AudioFile, Recognizer
from gtts import gTTS
from base64 import b64encode
from io import BytesIO
from pydub import AudioSegment
import re
import os
import random
import time
import pyttsx3
#from llama_index import GPTSimpleVectorIndex, download_loader
#set OPENAI_API_KEY=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


class Demo:
    def __init__(self):
        self.db = SQLDatabase.from_uri("sqlite:///./mydatabase.db")
        self.openai = OpenAI(temperature=0.7)
        self.chat = ChatOpenAI(temperature=0.7)
        self.chat_history = []
        self.prev_bot_response = ''
        self.question = ''
        persist_directory = './db'
        embedding = OpenAIEmbeddings()
        self.vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        self.llm = ChatOpenAI(model_name='gpt-3.5-turbo-0301', temperature=0, streaming=True, verbose=True)
        self.streaming_llm = ChatOpenAI(model_name='gpt-3.5-turbo-0301', streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=0)
        
    # def calender(self):
    #     GoogleCalendarReader = download_loader('GoogleCalendarReader')
    #     loader = GoogleCalendarReader()
    #     documents = loader.load_data()
    #     index = GPTSimpleVectorIndex(documents)
    #     index.query('When is Dr Dipesh available?')
        
    def db_query(self, query):
        chain = SQLDatabaseSequentialChain.from_llm(self.chat, database=self.db, verbose=True)
        result = chain.run(query)
        print(result)
        return result
        
    def create_doc(self, text):
        """Create documents from extracted json file to send to the model"""
        texts = text['Page'][0]['text']
        url = text['Page'][0]['url']
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.create_documents([texts], metadatas=[{"source": url}])
        return documents
        
    def chatbot(self, query):
        question = self.question + '' + query
        # Add current user question and previous bot response to chat history
        self.chat_history.append((question, self.prev_bot_response))
        question_generator = LLMChain(llm=self.llm, prompt=CONDENSE_QUESTION_PROMPT)
        doc_chain1 = load_qa_with_sources_chain(self.streaming_llm)
        
        qa = ChatVectorDBChain(vectorstore=self.vectordb, combine_docs_chain=doc_chain1, question_generator=question_generator)  #, return_source_documents=True       
        result = qa({"question": query, "chat_history": self.chat_history}) #return_only_outputs=True       
        bot_response = result["answer"] 

        # Update previous bot response with current response
        self.prev_bot_response = bot_response
        return bot_response
    
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

    def tts(self, text: str, language: str) -> object:
        """Converts text into audio object.
        Args:
            text (str): generated answer of bot
        Returns:
            object: text to speech object
        """
        return gTTS(text=text, lang=language, slow=False)
        # say out the response
        # engine = pyttsx3.init()
        # engine.setProperty('rate', 190)
        # engine.setProperty('voice', os.environ['VOICE'])
        # engine.say(output)
        # engine.startLoop()


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



    def tts_to_bytesio(self, tts_object: object) -> bytes:
        """Converts tts object to bytes.
        Args:
            tts_object (object): audio object obtained from gtts
        Returns:
            bytes: audio bytes
        """
        bytes_object = BytesIO()
        tts_object.write_to_fp(bytes_object)
        bytes_object.seek(0)
        return bytes_object.getvalue()

    def html_audio_autoplay(self, bytes: bytes) -> object:
        """Creates html object for autoplaying audio at gradio app.
        Args:
            bytes (bytes): audio bytes
        Returns:
            object: html object that provides audio autoplaying
        """
        b64 = b64encode(bytes).decode()
        html = f"""
        <audio controls autoplay>
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>
        """
        return html

# Initialize Chatbot class object
demo = Demo()
    
def pipeline(text=None,flag = None, audio=None , state="", statte="", stattte=""):
    if flag == 'chat':
        if audio:
            message = demo.recognize_speech_from_file(audio)
            print(message)
            state = demo.chatbot(message)
            audio = demo.tts(state, language="en-IN")
            print("Audio :", audio)
            bot_voice_bytes = demo.tts_to_bytesio(audio)
            html = demo.html_audio_autoplay(bot_voice_bytes)
            print("ANSWER :", state)
            return state, state, html
        if text: 
            # Record end time
            start_time1 = time.time()
            statte = demo.chatbot(text)
            print("STATTE  :", type(statte))
            end_time1 = time.time()
            # Calculate elapsed time
            elapsed_time1 = end_time1 - start_time1
            # Display elapsed time
            print("Chatbot: Elapsed time: {:.2f} seconds".format(elapsed_time1))
            #Record start time
            start_time2 = time.time() 
            # say out the response
            engine = pyttsx3.init()
            # engine.setProperty('rate', 190)
            # engine.setProperty('voice', os.environ['VOICE'])
            #engine.say(statte)
            # engine.startLoop() 
            engine.save_to_file(statte, "response.wav")
            engine.runAndWait()
            #audio = engine.getWaveData()
              
            # audio = demo.tts(statte, language="en-IN")
            # bot_voice_bytes = demo.tts_to_bytesio(audio)
            # html = demo.html_audio_autoplay(bot_voice_bytes)
            # Record end time
            end_time2 = time.time()
            # Calculate elapsed time
            elapsed_time2 = end_time2 - start_time2
            # Display elapsed time
            print("Total time elapsed: {:.2f} seconds".format(elapsed_time2))
            
            ####DIFFERENT######
            # import speech_recognition as sr
            # r = sr.Recognizer()
            # with sr.AudioFile("response.wav") as source:
            #     audio_data = r.record(source)

            return statte, statte, None, "response.wav"
    elif flag == 'db':
        stattte = demo.db_query(text)
        return stattte, stattte, None, None
     
if __name__ == "__main__":
    #gr.outputs.Audio()
    #output = gr.outputs.Audio("response.wav")
    gr.Interface(
        fn=pipeline,
        inputs=["text", "text", gr.inputs.Audio(source="microphone", type="filepath"), "state"], 
        outputs=["text", "state", "html", gr.outputs.Audio(type="filepath")]).launch(debug=True)
        

