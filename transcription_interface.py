import gradio as gr
from gradio_client import Client
import numpy as np
import whisper

# Define a function that loads and returns the Whisper model based on user selection
def load_whisper_model(model_name):
    return whisper.load_model(model_name)

paramfp16=False # Set to True if you want to use fp16 precision on GPU

def transcribe(audio):
    model = whisper.load_model("base")
    result = model.transcribe(audio,fp16=paramfp16)
    print(result["text"])
    return result["text"]

def processAudio(audio1,audio2,choiceTranslate):
    model = whisper.load_model("base")

    if audio1 is None and audio2 is None:
        return "No audio inputs were provided."
    elif audio1 is None:
        # Process only the second audio input
        # Your audio processing code here
        # For this example, we'll just return the second audio input
        audioOk = audio2
    elif audio2 is None:
        # Process only the first audio input
        # Your audio processing code here
        # For this example, we'll just return the first audio input
        audioOk = audio1
    else: 
        audioOk = audio1
    result = model.transcribe(audioOk,fp16=paramfp16)
    print(result["text"])
    return result["text"]

iface = gr.Interface(
    fn=processAudio, 
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="Record Audio", show_label=True),
        gr.Audio(sources=["upload"], type="filepath", label="Upload Audio", show_label=True),
    ],
    gr.Dropdown(label="Choose Whisper Model", choices=["tiny", "base", "small", "medium", "large"], default="base")
    outputs="text",
    title="Whisper-based transcription app",
    description="Record your speech via microphone or upload an audio file and press the Submit button to transcribe it into text. Please, note that the size of the audio file should be less than 25 MB."
)

iface.launch()