from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.readers.file import FlatReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
#from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
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
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import textwrap
import tempfile

with open('credentials.json', 'r', encoding='utf-8') as f:
    credentials = json.load(f)
os.environ['OPENAI_API_KEY'] = credentials['openai_api_key']

def handle_upload_old(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:    # Read the content directly from the file-like object
            file_content = f.read()
            wrapped_text = textwrap.fill(file_content, width=120)
            displayed_content = wrapped_text
            file_name = os.path.basename(file_path)
            file_name_display=file_name# Get the filename from the path
            return file_content, file_name
    except Exception as e:
        return f"Error processing file: {str(e)}", ""

def handle_upload(file_info):
    # Check if file_info is None or has another issue
    if file_info is None:
        return "No file uploaded."
    
    try:
        file_name, file_data, *_ = file_info  # Using *_ to ignore extra values
    except ValueError as e:
        return f"Error processing file upload: {e}"
    
    # Assuming file_data is a path or file-like object here
    try:
        # If file_data is a file path
        with open(file_info, 'r', encoding='utf-8') as file:
            file_content = file.read()
    except TypeError:
        # Assuming it's a file-like object if the above fails (adjust logic as needed)
        file_content = file_info.read().decode('utf-8')
            
    # Use textwrap to format the content.
    wrapped_text = textwrap.fill(file_content, width=120)

    # Prepare the variable for display.
    displayed_content = wrapped_text  # Modify the slice as needed.
    
    # Return the prepared content.
    return displayed_content, file_name  # Now returning the prepped and wrapped content for display

def ingest_uploaded_files(file_content):
    f = io.StringIO()  # Create a string buffer
    
   #create a pipeline with transformations
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=200, chunk_overlap=10),
            TitleExtractor(nodes=3),
            OpenAIEmbedding(),
        ]
    )
    # run the pipeline and process the documents directly from file_content variable
    nodes = pipeline.run(documents=[Document(text=file_content)])

    # build index
    index = VectorStoreIndex(nodes)
    retriever = index.as_retriever(similarity_top_k=3)

    # Detail on how each node was processed
    with redirect_stdout(f):
        print("Starting pipeline processing...")
        
        # Detail on how each node was processed
        if nodes:
            for i, node in enumerate(nodes, start=1):
                print(f"\nNode {i}:")
                if hasattr(node, 'get_content'):
                    print(f"  Content: {node.get_content()}")
                else:
                    print("  Content could not be retrieved; 'get_content' method missing.")

                if hasattr(node, 'metadata'):
                    # Assuming the node has a 'metadata' attribute containing processed info
                    print(f"  Metadata: {node.get_content(metadata_mode="all")}")
                else:
                    print("  Metadata could not be retrieved; 'metadata' attribute missing.")
        
                # Add any other attributes you might want to print out
                # Example:
                # if hasattr(node, 'title'):
                #     print(f"  Title: {node.title}")

            # Print a summary
            print(f"\nTotal nodes created and processed: {len(nodes)}.")
        else:
            print("No nodes were created during processing.")

        # Assuming the final statement summarizing the process, for example:
        print(f"Files processed and indexed. {len(nodes)} nodes (chunks) created.")
        
    output = f.getvalue()  # Retrieve string from StringIO buffer
    return output

def ask_question(question, file_content):  
    if text != "": # Check if a question has been asked
         # Your implementation for asking the question using LLM and Llamaindex...
         return "Your Answer"
    else: 
        return "" # Return an empty string to hide Textbox in UI.

# Gradio Interface in two tabs one for upload the other for asking questions
with gr.Blocks(title="Ask Questions to text files") as app:
    gr.Markdown("# Ask Questions to text files\nUpload a text file and ask questions to the file. A LLM will provide the answers.")
    file_content_display = gr.Textbox(value='', label="Content of the file", visible=True, interactive=False)
    file_name_display = gr.Textbox(value='', label="File Name", visible=True)  # Define this component to display the file name
    
    with gr.TabItem("Step.1 Upload Document"):
        file = gr.File(file_types=['.txt', '.md'], label='Choose a file to upload', type="filepath")
        btn_upload = gr.Button('Upload')
        btn_upload.click(handle_upload_old, inputs=file, outputs=[file_content_display, file_name_display])
    
    with gr.TabItem("Step.2 Data ingestion"):
        nlp_ingestion_result = gr.Textbox(placeholder="Status of ingestion", label='Processing files')
        btn_process_nlp = gr.Button('Ingest file')
        btn_process_nlp.click(ingest_uploaded_files, inputs=file_content_display, outputs=nlp_ingestion_result)
        
    with gr.TabItem("Step.3 Ask Question"):
        question_box = gr.Textbox(placeholder="Enter your question here", label='Question')
        answer_box = gr.Textbox(placeholder="Your answers will appear here...", label='Answer', interactive=False)
        btn_ask = gr.Button('Ask')
        btn_ask.click(ask_question, inputs=[question_box, file_content_display], outputs=answer_box)

# Launch the application
if __name__ == "__main__":
    app.launch()