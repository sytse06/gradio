# Import necessary libraries
import gradio as gr
from datetime import timedelta
import numpy as np
from PIL import Image
import io
import json
import os

# Initialize the Whisper transcription API interface instance
interface = gr.Interface(f"whisper.SpeechClient", inputs="audio", outputs="text")

# Define the models available for transcription using Gradio
models = {
    "en-us": {
        "params": "whisper.en-us",
        "description": "This model is trained on a dataset of English audio recordings and can be used to transcribe speech in American English."
    },
    "de-de": {
        "params": "whisper.de-de",
        "description": "Dieser Modell ist auf einem Datensatz von deutschen Audio-Aufzeichnungen trainiert und kann für die Sprachsynthese in Deutsch verwendet werden."
    }
}

# Define the output formats available for transcription using Gradio
output_formats = {
    "text/plain": {
        "description": "This is a plain text representation of the transcribed speech.",
        "format": "text"
    },
    "application/json": {
        "description": "This is a JSON object containing information about the transcription, including the recognized words and their timestamps.",
        "format": "json"
    }
}

# Define the Gradio interface for handling user input
app = gr.Interface(inputs="audio", outputs=output_formats["text/plain"], parameters=models)

# Define a function to transcribe audio using the selected model and output format
def transcribe(audio, model, output_format):
    # Transcribe the entire audio and retrieve the transcribed result
    result = interface.run_async(audio, params={"model": model})
    
    # Get the transcription result from the model
    transcription = result["results"][0]["alternatives"][0]["transcript"]
    
    return {"transcription": transcription}

# Define a function to format timestamp information for each segment
def format_timestamps(segments):
    # Initialize start time as None
    start = None
    
    # Initialize end time as None
    end = None
    
    # Initialize duration as None
    duration = None
    
    # Iterate over the segments and update the start, end, and duration if necessary
    for segment in segments:
        if start is None:
            start = segment["start"]
        
        if end is None:
            end = segment["end"]
            
        if duration is None:
            duration = segment["duration"]
    
    return {"start": start, "end": end, "duration": duration}

# Define a function to get the transcription result in the specified format
def get_transcription(audio, model, output_format):
    # Transcribe the audio using the selected model and output format
    result = interface.run(audio, params={"model": model})
    
    # Get the transcription result in the specified format
    transcription = result["results"][0]["alternatives"][0]["transcript"]
    
    # Format the timestamp information for each segment
    segments = result["results"][0]["alternatives"][0]["segments"]
    formatted_timestamps = format_timestamps(segments)
    
    return {"transcription": transcription, "formatted_timestamps": formatted_timestamps}

# Launch the demo
demo.launch()