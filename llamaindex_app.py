from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.readers.file import FlatReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
#from llama_index.core.postprocessor import SimilarityPostprocessor
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

def handle_upload(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:    # Read the content directly from the file-like object
        content = f.read()
        
    wrapped_text = textwrap.fill(content, width=120)  # Wrap your text to fit into a specified width. By default, it wraps at 70 characters.
    
    return wrapped_text

def ingest_uploaded_files(file_path):
    documents = []
    all_nodes = []

    for uploaded_file in uploaded_files:
        # Gradio passes a tuple for each file consisting of (filename, file)
        original_filename, file_obj = uploaded_file
        
        # Create a secure temporary file to mirror the uploaded file, preserving the original extension
        suffix = Path(original_filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_path).suffix) as tmp_file:
            # Write the uploaded file's content to the temp file
                    with open(file_path, 'rb') as f:
                tmp_file.write(f.read())
            tmp_file_path = tmp_file.name
            
    # Implement your ingestion logic here
        # Use FlatReader to load the document from the temp file
        doc = FlatReader().load_data(tmp_file_path)
        documents.extend(doc)  # Append all loaded documents
        
        # Cleanup the temp file
        tmp_file_path.unlink()

    # Split content of files into chunks
    text_splitter = SentenceSplitter(chunk_size=200, chunk_overlap=10, include_metadata=True)
    nodes = text_splitter.get_nodes_from_documents(documents=documents)
    all_nodes.extend(nodes)
    
    # Create an index to process queries
    index = VectorStoreIndex.from_documents(nodes=nodes, embed_model=OpenAIEmbedding())
    retriever = index.as_retriever(similarity_top_k=3)
    
    # Return a message with detailed information
    return f"Processed file located at: {tmp_file_path}"
    return f"Files indexed successfully with metadata. Number of nodes (chunks) created: {len(nodes)}."

def ask_question(question, file_content):  
    if text != "": # Check if a question has been asked
         # Your implementation for asking the question using LLM and Llamaindex...
         return "Your Answer"
    else: 
        return "" # Return an empty string to hide Textbox in UI.

# Gradio Interface in two tabs one for upload the other for asking questions
with gr.Blocks(title="Ask Questions to text files") as app:
    gr.Markdown("""
# Ask Questions to text files 
Upload a text file and ask questions to the file. A LLM will provide the answers.""")
    file_content_display = gr.Textbox(value='', label="Content of the file", visible=True, interactive=False)

    with gr.TabItem(label="Step.1 Upload Document"):
        file = gr.File(file_types=['.txt', '.md'], label='Choose a file to upload')
        btn_upload = gr.Button('Upload')
        btn_upload.click(fn=handle_upload, inputs=[uploaded_file], outputs=file_content_display)
    
    with gr.TabItem(label="Step.2 Data ingestion"):
        nlp_ingestion_result = gr.Textbox(placeholder="Status ingestion", label='Processing files')
        #nlp_result_output = gr.Textbox(label="NLP Processing Result")
        btn_process_nlp = gr.Button('Ingest file')
        btn_process_nlp.click(fn=ingest_uploaded_files, inputs=[uploaded_file], outputs=[nlp_ingestion_result])
        
    with gr.TabItem(label="Step.3 Ask Question"):
        question_box = gr.Textbox(placeholder="Enter your question here", label='Question')
        answer_box = gr.Textbox(placeholder="Your answers will appear here...", label='Answer', interactive=False)
        btn_ask = gr.Button('Ask')
        # Pass both the question and the content of the file to the function
        btn_ask.click(fn=ask_question, inputs=[question_box, file_content_display], outputs=answer_box)

# Launch the application
if __name__ == "__main__":
    app.launch()