from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.schema import AIMessage, HumanMessage
import openai
import gradio as gr
import os

#Instantiate config class for open source LLM
class LLMConfiguration:
    def __init__(self, api_base, api_key, model_name, port=443):
        self.api_base = api_base
        self.api_key = 'Null'
        self.model_name = model_name
        self.port = port

# Define the get_llm function here
def get_llm(config):
    api_base = config["api_base"]
    api_key = config["api_key"]
    model_name = config["model_name"]
    port = config.get("port", 443)
    
    if model_name == "gemma:7b":
        return ChatOpenAI(api_base=api_base, api_key=api_key, model_name=model_name, port=port)
    elif model_name == "Mixtral-7B":
        return ChatOpenAI(api_base=api_base, api_key=api_key, model_name=model_name, port=port)
    else:
        # Handle any other models if needed
        pass

# Define the predict function here
def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = llm.predict(history_langchain_format)
    return gpt_response.content

gr.ChatInterface(predict).launch()