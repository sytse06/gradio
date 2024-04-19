from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.readers.file import FlatReader
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings, Document, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
#from IPython.display import Markdown, display
import chromadb
import gradio as gr
import re
import os
import json
from pathlib import Path

with open('credentials.json', 'r', encoding='utf-8') as f:
    credentials = json.load(f)
os.environ['OPENAI_API_KEY'] = credentials['openai_api_key']

def handle_upload(file):

    return open(file, 'r').read()   # assuming the returned value is content of uploaded files

def ask_question(text):  
    if text != "": # Check if a question has been asked
         # Your implementation for asking the question using LLM and Llamaindex...
         return "Your Answer"
    else: 
        return "" # Return an empty string to hide Textbox in UI.

# Create Gradio interface
with gr.TabbedInterface(['Upload Document', 'Ask Question']) as ui:
    with ui.tabItem("Upload Document"):
        file = gr.File(file_types=['.txt'], label='Choose a text file to upload')
        btn_upload = gr.Button('Submit').style(margin=False)
        
    with ui.tabItem("Ask Question"):
        question_box = gr.Textbox(placeholder="Enter your question here", label='Question')
        answer_box = gr.Textbox(placeholder="Your answers will appear here...", label='Answer', disabled=True)
        btn_ask = gr.Button('Ask').style(margin=False)
        
    # Define event handlers
    btn_upload.click(handle_upload, [file], iface2)  
    ui.close(iface1).then(btn_ask, answer_box, lambda x: f"You asked: {x}", show=True)   
    btn_ask.click(lambda x: f"You asked: {x}", question_box, answer_box)  # Define event handlers
    
if __name__ == "__main__":
    ui.launch()