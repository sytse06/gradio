import gradio as gr
from langchain_community.document_loaders import WebBaseLoader 
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
#from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import CharacterTextSplitter

def process_input(urls, question):
    model_local = ChatOllama(model="mistral")
    
    # Convert string of URLs to list (Ensure this logic is correctly implemented)
    urls_list = urls.split('\n')  # Assuming 'urls' is the input string containing URLs separated by new lines
    fixed_urls_list = [url.strip() for url in urls_list if url.strip()]  # Removes any empty strings

    if not fixed_urls_list:
        return "No URLs provided. Please enter valid URLs separated by new lines."

    docs = [WebBaseLoader(url).load() for url in fixed_urls_list]
    docs_list = [item for sublist in docs for item in sublist]
    if not docs_list:
        return "Failed to load documents from the provided URLs. Please check the URLs and try again."
    
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs_list)
    if not doc_splits:
        return "Document processing resulted in no valid text chunks. Please check the content of the provided URLs."
    embedding_function = OllamaEmbeddings(model='nomic-embed-text')
    #embedding_function = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embedding_function
    )
    retriever = vectorstore.as_retriever()

    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )
    return after_rag_chain.invoke(question)

# Define Gradio interface
iface = gr.Interface(fn=process_input,
                     inputs=[gr.Textbox(label="Enter URLs separated by new lines"), gr.Textbox(label="Question")],
                     outputs="text",
                     title="Document Query with Ollama",
                     description="Enter URLs and a question to query the documents.")
iface.launch()