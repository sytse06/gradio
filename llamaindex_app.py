
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import Document
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

def upload_file(files):

    file_paths = [file.name for file in files]

    return file_paths

with gr.Blocks() as demo:

    file_output = gr.File()

    upload_button = gr.UploadButton("Click to Upload a File", file_types=["text"], file_count="single")

    upload_button.upload(upload_file, upload_button, file_output)

def handle_upload(files):
    processed_files = []  # List to store processed text files
    
    if files is not None:
        for file in files:
            with open(file.name, 'r') as f:
                content = f.read()  # Read content of the file
            
            # Process content here or pass it directly to Document constructor
            processed_content = process_content(content)
            
            # Create Document object directly from the processed content
            document = Document(text=processed_content)
            
            # Append Document object to the list
            processed_files.append(document)
        
        return "Files uploaded and processed successfully."
    else:
        return "No file was uploaded."
    
# Create Document objects directly from the processed text files
documents = [Document(text=content) for content in processed_files]

from llama_index.core.node_parser import SentenceSplitter

text_splitter = SentenceSplitter(chunk_size=200, chunk_overlap=10)
nodes = text_splitter.get_nodes_from_documents(documents=documents)

from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding

chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("tes1233t")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=OpenAIEmbedding())

retriever = index.as_retriever()

retriever.retrieve("How long does it take to prepare a pizza")

# llm = OpenAI(model="gpt-3.5-turbo")

# query_engine = index.as_query_engine(llm=llm)

Settings.llm = OpenAI(model="gpt-3.5-turbo")

query_engine.query("How long does it take to prepare a pizza")

from llama_index.core import PromptTemplate


new_summary_tmpl_str = (
    "You always say 'Hello my friend' at the beginning of your answer. Below you find data from a database\n"
    "{context_str}\n"
    "Take that context and try to answer the question with it."
    "Query: {query_str}\n"
    "Answer: "
)
new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)

query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": new_summary_tmpl}
)

prompts_dict = query_engine.get_prompts()
print(prompts_dict)
query_engine.query("How long does it take to prepare a pizza")

# Create Gradio interface
iface = gr.Interface(fn=handle_upload, inputs=gr.inputs.Upload(file_count='multiple'), outputs="text")