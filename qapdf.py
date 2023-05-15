import os 
import sys 
import getpass
from dotenv import load_dotenv
import gradio as gr 
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper, StorageContext, load_index_from_storage
import pyttsx3
from langchain.document_loaders import PyPDFLoader
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0301", max_tokens=num_outputs))

    docs = SimpleDirectoryReader(directory_path).load_data()

    index = GPTVectorStoreIndex.from_documents(docs, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.storage_context.persist(persist_dir="index")

    return index

def qabot(input_text):
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="index")

    # load index
    index = load_index_from_storage(storage_context)
    
    query_engine = index.as_query_engine()
    response = query_engine.query(input_text)
    print("Response :", response)
    engine = pyttsx3.init() 
    engine.save_to_file(response, "pdf.wav")
    engine.runAndWait()
    return response, "pdf.wav"


#input_pdf_path = gr.inputs.Textbox(label="Enter the PDF file path")

iface = gr.Interface(fn=qabot, inputs=gr.inputs.Textbox(lines=7, label='Enter your query'),outputs=["text", gr.outputs.Audio(type="filepath")], title="Custom-trained QA Application")
index = construct_index("docs")
iface.launch(share=True)
