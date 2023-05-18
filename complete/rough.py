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
import textwrap
import re
import asyncio
import aiohttp
from num2words import num2words
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
        self.loader = PyMuPDFLoader("docs/report1.pdf")

    
    def replace_numbers_with_words(self, filename):
        def conv_num(match):
            num = int(match.group())
            return num2words(num)

        # Read the text file
        with open(filename, 'r') as file:
            text = file.read()

        # Replace numbers with words
        replaced_text = re.sub(r'\b\d+\b', conv_num, text)

        # Write the updated text back to the file
        with open(filename, 'w') as file:
            file.write(replaced_text)

        print("Numbers replaced with words in", filename)
        
    def chatbot(self, query):
        data = self.loader.load()
        for i in range(0, len(data)):
            self.numbers_to_words(data[i].page_content)
        print("DATA :", data[1].page_content)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(data)
        db = Chroma.from_documents(docs, self.embedding, persist_directory=self.persist_directory1)
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

async def convert_text_to_speech(sentence, session):
    # Make the API call to convert the sentence to speech
    async with session.get('https://your_tts_api.com', params={'text': sentence}) as response:
        audio = await response.read()

    return audio

async def convert_list_to_speech(sentences):
    # Create an aiohttp session
    async with aiohttp.ClientSession() as session:
        # Prepare a list to store the speech audio
        audio_list = []

        # Create a list of tasks for parallel execution
        tasks = []
        for sentence in sentences:
            task = asyncio.ensure_future(convert_text_to_speech(sentence, session))
            tasks.append(task)

        # Execute the tasks concurrently
        audio_list = await asyncio.gather(*tasks)

    # Concatenate the audio into a single file or buffer
    audio = b''.join(audio_list)

    # Return the audio in case it's needed for further processing
    return audio


# if __name__ == '__main__':
#     colors = ['red', 'blue', 'green']  # ...
#     # Either take colors from stdin or make some default here
#     asyncio.run(main(colors))

     
if __name__ == "__main__":
    gr.Interface(
        fn=split_streaming_response,
        inputs="text",
        outputs="text").launch(debug=True) #gr.outputs.Audio(type="filepath")
        

