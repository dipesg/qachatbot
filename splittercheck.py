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
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
import gradio as gr
import speech_recognition as sr
import asyncio
import pyttsx3
import re
from tt import call_asr, call_tts_nepali, trans
###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

class Demo:
    def __init__(self):
        self.chat_history = []
        self.prev_bot_response = ''
        self.question = ''
        self.persist_directory2 = './split'
        self.embedding = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name='gpt-3.5-turbo-0301', temperature=0, streaming=True, verbose=True)
        self.streaming_llm = ChatOpenAI(model_name='gpt-3.5-turbo-0301', streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=0)
        loader = PyMuPDFLoader("docs/report1.pdf")
        data = loader.load()
        text_splitter = CharacterTextSplitter()
        docs = text_splitter.split_documents(data)
        db = Chroma.from_documents(docs, self.embedding, persist_directory=self.persist_directory2)
        
        
    def chatbot(self, query):
        question = self.question + '' + query
        # Add current user question and previous bot response to chat history
        self.chat_history.append((question, self.prev_bot_response))
        self.vectordb = Chroma(persist_directory=self.persist_directory2, embedding_function=self.embedding)
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

async def call_gtts_api(text, index):
    loop = asyncio.get_running_loop()
    tts = call_tts_nepali(text)#gTTS(text=text, lang='en')
    tts.save(f'output_{index}.mp3')

    with open(f'output_{index}.mp3', 'rb') as f:
        audio_content = f.read()

    return audio_content

async def transcribe_audio(file_path):
        try:
            text = call_asr(file_path)
            print("Transcription: ", text)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

        return text

async def main(text):
    sentences = split_streaming_response(text)
    audio_tasks = []
    for i, sentence in enumerate(sentences, start=1):
        audio_task = asyncio.create_task(call_gtts_api(sentence, i))
        audio_tasks.append(audio_task)

    audio_contents = await asyncio.gather(*audio_tasks)

    # transcript_tasks = []
    # for audio_content in audio_contents:
    #     transcript_task = asyncio.create_task(transcribe_audio(audio_content))
    #     transcript_tasks.append(transcript_task)

    # transcripts = await asyncio.gather(*transcript_tasks)
    return audio_contents

# Define your Gradio interface function
async def process_text(audio=None, text=None):
    if audio:
        message = demo.recognize_speech_from_file(audio)#transcribe_audio(audio)
        # Process the audio contents as needed
        output = demo.chatbot(message)
        aud = await main(output)
        return "Processed output"
    else:
        message = text
        audio_contents = await main(message)
        # Process the audio contents as needed
        # ...

        return "Processed output"

    
# def pipeline(text=None, audio=None , state="", statte=""):
#     if audio:
#         message = demo.recognize_speech_from_file(audio)
#         engine = pyttsx3.init()
#         engine.save_to_file(statte, "response.wav")
#         engine.runAndWait()
#         state = demo.chatbot(message)
#         return state, state,"response.wav"
#     if text: 
#         statte = demo.chatbot(text)
#         engine = pyttsx3.init()
#         engine.save_to_file(statte, "response.wav")
#         engine.runAndWait()
#         return statte, statte,"response.wav"
     
if __name__ == "__main__":
    # Create your Gradio interface
    # Define your Gradio interface components
    input_text = gr.inputs.Textbox(label="Input Text")
    input_aud = gr.inputs.Audio(source="microphone", type="filepath")
    output_text = gr.outputs.Textbox(label="Output Text")
    output_aud = gr.outputs.Audio(type="filepath")
    gr.Interface(fn=process_text, inputs=[input_aud,input_text], outputs=output_text).launch()
    # gr.Interface(
    #     fn=pipeline,
    #     inputs=["text", gr.inputs.Audio(source="microphone", type="filepath"), "state"], 
    #     outputs=["text", "state", gr.outputs.Audio(type="filepath")]).launch(debug=True)