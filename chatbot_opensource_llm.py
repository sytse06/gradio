from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.schema import AIMessage, HumanMessage
import openai
import gradio as gr
import os

import os
import json
with open('/Users/sytsevanderschaaf/Documents/Dev/credentials.json', 'r', encoding='utf-8') as f:
    credentials = json.load(f)
os.environ['OPENAI_API_KEY'] = credentials['openai_api_key']

llm_ollama = ChatOpenAI(
    openai_api_base="http://localhost:11434",
    openai_api_key="Null",                 
    model_name="Mistral"
)

llm_lmstudio = ChatOpenAI(
    openai_api_base="http://localhost:4321/v1",
    openai_api_key="Null",                 
    model_name="Mistral"
)

def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = llm_lmstudio(history_langchain_format)
    return gpt_response.content

gr.ChatInterface(predict).launch()