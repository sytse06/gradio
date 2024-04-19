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

# Function to handle the upload and processing of files, including metadata
def handle_upload(file_paths):
    documents = []
    for file_path in file_paths:
        # Use FlatReader to load the document
        doc = FlatReader().load_data(Path(file_path))
        documents.extend(doc)  # Append all loaded documents
    
    # Parse documents into nodes using SimpleFileNodeParser
    parser = SimpleFileNodeParser(include_metadata=True)
    all_nodes = []
    for document in documents:
        nodes = parser.get_nodes_from_documents([document])
        all_nodes.extend(nodes)
    return f"Processed {len(all_nodes)} nodes from uploaded files."

# Function to process and index documents
def index_documents(documents):
    text_splitter = SentenceSplitter(chunk_size=200, chunk_overlap=10, include_metadata=True)
    
    # Gather all nodes from the documents
    nodes = text_splitter.get_nodes_from_documents(documents=documents)
    
    # Detail about chunks and nodes
    number_of_chunks = len(nodes)  # Total number of chunks created from all documents
    
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("test_collection2")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=OpenAIEmbedding())
    retriever = index.as_retriever()
    
    # Return a message with detailed information
    return f"Files indexed successfully with metadata. Number of nodes (chunks) created: {number_of_chunks}."

llm = OpenAI(model="gpt-3.5-turbo")

new_summary_tmpl_str = (
    "You always say 'Hello my friend' at the beginning of your answer. Below you find data from a database\n"
    "{context_str}\n"
    "Take that context and try to answer the question with it."
    "Query: {query_str}\n"
    "Answer: "
)
new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)

#retriever.update_prompts({"response_synthesizer:text_qa_template": new_summary_tmpl})

#prompts_dict = query_engine.get_prompts()
#print(prompts_dict)
#query_engine.query("How long does it take to prepare a pizza")

# Create Gradio interface
iface = gr.Interface(
    fn=handle_upload, 
    inputs=[
        gr.File(
            file_types=['txt', 'md', 'json'],
            type="filepath",
            label="Upload text files",
            show_label=True,
            interactive=True,
            file_count="multiple",
            ),
        gr.Textbox(label="Enter your question", visible=True, placeholder="Ask a question..."),
        ], 
    outputs="text"
)

if __name__ == "__main__":
    iface.launch()