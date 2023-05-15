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
import time
import pyttsx3
from dotenv import load_dotenv
import gradio as gr 
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
import pyttsx3
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyMuPDFLoader
#from llama_index import GPTSimpleVectorIndex, download_loader
#set OPENAI_API_KEY=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


class Demo:
    def __init__(self):
        self.chat_history = []
        self.prev_bot_response = ''
        self.question = ''
        self.persist_directory1 = './pdf_db'
        self.embedding = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name='gpt-3.5-turbo-0301', temperature=0, streaming=True, verbose=True)
        self.streaming_llm = ChatOpenAI(model_name='gpt-3.5-turbo-0301', streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=0)
        loader = PyMuPDFLoader("docs/geoprod.pdf")
        data = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(data)
        db = Chroma.from_documents(docs, self.embedding, persist_directory=self.persist_directory1)
        
        
    def chatbot(self, query):
        question = self.question + '' + query
        # Add current user question and previous bot response to chat history
        self.chat_history.append((question, self.prev_bot_response))
        self.vectordb = Chroma(persist_directory=self.persist_directory1, embedding_function=self.embedding)
        question_generator = LLMChain(llm=self.llm, prompt=CONDENSE_QUESTION_PROMPT)
        doc_chain1 = load_qa_chain(self.streaming_llm, prompt=QA_PROMPT)
        qa = ChatVectorDBChain(vectorstore=self.vectordb, combine_docs_chain=doc_chain1, question_generator=question_generator)  #, return_source_documents=True
        result = qa({"question": query, "chat_history": self.chat_history})
        bot_response = result["answer"] 
        # Update previous bot response with current response
        self.prev_bot_response = bot_response
        return bot_response

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
    
def pipeline(text=None, audio=None , state="", statte=""):
    if audio:
        message = demo.recognize_speech_from_file(audio)
        engine = pyttsx3.init()
        engine.save_to_file(statte, "response.wav")
        engine.runAndWait()
        state = demo.chatbot(message)
        return state, state,"response.wav"
    if text: 
        statte = demo.chatbot(text)
        engine = pyttsx3.init()
        engine.save_to_file(statte, "response.wav")
        engine.runAndWait()
        return statte, statte,"response.wav"
     
if __name__ == "__main__":
    gr.Interface(
        fn=pipeline,
        inputs=["text", gr.inputs.Audio(source="microphone", type="filepath"), "state"], 
        outputs=["text", "state", gr.outputs.Audio(type="filepath")]).launch(debug=True)
        

