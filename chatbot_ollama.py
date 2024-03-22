from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama
from langchain.schema import AIMessage, HumanMessage
import gradio as gr
import os

import json

ollama = Ollama(base_url='http://localhost:11434',
model="deepseek-coder:6.7b")

def chat(prompt):
    return ollama([HumanMessage(content=prompt)])[0].content  # get only the first response's content

def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))

    gpt_response = ollama(history_langchain_format)
    return gpt_response
gr.ChatInterface(predict).launch()