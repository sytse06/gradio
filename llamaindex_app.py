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
import textwrap

with open('credentials.json', 'r', encoding='utf-8') as f:
    credentials = json.load(f)
os.environ['OPENAI_API_KEY'] = credentials['openai_api_key']

def handle_upload(file):
    with open(file, 'r') as f:   # Read the content directly from the file-like object
        content = f.read()
        
    wrapped_text = textwrap.fill(content, width=80)  # Wrap your text to fit into a specified width. By default, it wraps at 70 characters.
    
    return wrapped_text

def ask_question(question, file_content):  
    if text != "": # Check if a question has been asked
         # Your implementation for asking the question using LLM and Llamaindex...
         return "Your Answer"
    else: 
        return "" # Return an empty string to hide Textbox in UI.

# Gradio Interface in two tabs one for upload the other for asking questions
with gr.Blocks() as app:
    file_content_display = gr.Textbox(value='', label="Content of the file", visible=True, interactive=False)

    with gr.TabItem(label="Upload Document"):
        file = gr.File(file_types=['.txt', '.md'], label='Choose a file to upload')
        btn_upload = gr.Button('Submit')
        btn_upload.click(handle_upload, inputs=file, outputs=file_content_display)
        
    with gr.TabItem(label="Ask Question"):
        question_box = gr.Textbox(placeholder="Enter your question here", label='Question')
        answer_box = gr.Textbox(placeholder="Your answers will appear here...", label='Answer', interactive=False)
        btn_ask = gr.Button('Ask')
        # Pass both the question and the content of the file to the function
        btn_ask.click(ask_question, inputs=[question_box, file_content_display], outputs=answer_box)

# Launch the application
if __name__ == "__main__":
    app.launch()