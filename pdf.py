import gradio as gr
from langchain import OpenAI, PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
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

llm = OpenAI(temperature=0)
embedding = OpenAIEmbeddings()
persist_directory = "./sum_qa"
chat_history = []
prev_bot_response = ''
question = ''
# embedding = OpenAIEmbeddings()
llm = ChatOpenAI(model_name='gpt-3.5-turbo-0301', temperature=0, streaming=True, verbose=True)
streaming_llm = ChatOpenAI(model_name='gpt-3.5-turbo-0301', streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=0)

def summarize_pdf(pdf_file_path, query=None):
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load_and_split()
    db = Chroma.from_documents(docs, embedding, persist_directory=persist_directory)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)
    while True:
        question = question + '' + query
        # Add current user question and previous bot response to chat history
        chat_history.append((question, prev_bot_response))
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
        doc_chain1 = load_qa_with_sources_chain(streaming_llm)
        qa = ChatVectorDBChain(vectorstore=vectordb, combine_docs_chain=doc_chain1, question_generator=question_generator)  #, return_source_documents=True       
        result = qa({"question": query, "chat_history": chat_history}) #return_only_outputs=True       
        bot_response = result["answer"] 
        # Update previous bot response with current response
        prev_bot_response = bot_response

    
    # if custom_prompt!="":
    #     prompt_template = custom_prompt + """

    #     {text}

    #     SUMMARY:"""
    #     PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    #     chain = load_summarize_chain(llm, chain_type="map_reduce", 
    #                                 map_prompt=PROMPT, combine_prompt=PROMPT)
    #     custom_summary = chain({"input_documents": docs},return_only_outputs=True)["output_text"]
    # else:
    #     custom_summary = ""
    
        return summary, bot_response


def custom_summary(pdf_file_path, custom_prompt):
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load_and_split()
    prompt_template = custom_prompt + """

    {text}

    SUMMARY:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="map_reduce", 
                                map_prompt=PROMPT, combine_prompt=PROMPT)
    summary_output = chain({"input_documents": docs},return_only_outputs=True)["output_text"]
    
    return summary_output
    

def main():
    input_pdf_path = gr.inputs.Textbox(label="Enter the PDF file path")
    #input_custom_prompt = gr.inputs.Textbox(label="Enter your custom prompt")
    input = gr.inputs.Textbox(label="Enter your Query")
    output_summary = gr.outputs.Textbox(label="Summary")
    #output_custom_summary = gr.outputs.Textbox(label="Custom Summary")
    result = gr.outputs.Textbox(label="ANswer")

    iface = gr.Interface(
        fn=summarize_pdf,
        inputs=[input_pdf_path, input],
        outputs=[output_summary, result],
        title="PDF Summarizer",
        description="Enter the path to a PDF file and get its summary.",
    )
    
    iface.launch()

if __name__ == "__main__":
    main()