from flask import Flask, render_template, request
import gradio as gr
import numpy as np
import whisper
from datetime import timedelta
from gradio_audio_interface.models.whisper_model import WhisperModel, WhisperModelParameters
from gradio_audio_interface.models.whisper_model.types import Inputs, Outputs
from gradio_audio_interface.models.whisper_model.utils import get_whisper_model_params

app = Flask(__name__)

# Select the available Whisper models from a list to load it for transcription
whisper_models = {
    "tiny": {
        "size": 39, # in MB
        "params": "en-US", # language model parameters
        "en_only": True, # only supports English language
        "multilingual": False
    },
    "base": {
        "size": 74, # in MB
        "params": "en-US", # language model parameters
        "en_only": True, # only supports English language
        "multilingual": False
    },
    "small": {
        "size": 244, # in MB
        "params": "en-US", # language model parameters
        "en_only": True, # only supports English language
        "multilingual": False
    },
    "medium": {
        "size": 769, # in MB
        "params": "en-US", # language model parameters
        "en_only": True, # only supports English language
        "multilingual": False
    },
    "large": {
        "size": 1550, # in MB
        "params": None, # no specific language model parameters
        "en_only": False, # can recognize multiple languages
        "multilingual": True
    },
    "large-v2": {
        "size": 1550, # in MB
        "params": None, # no specific language model parameters
        "en_only": False, # can recognize multiple languages
        "multilingual": True
    },
    "large-v3": {
        "size": 1550, # in MB
        "params": None, # no specific language model parameters
        "en_only": False, # can recognize multiple languages
        "multilingual": True
    }
}

# Load the available output formats
output_formats = {
    "text/plain": {"description": "Plain text format"},
    "text/tab-separated-values": {"description": "Tab-separated values format"},
    "application/json": {"description": "JSON format"}
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
        """
    Upload an audio file, transcribe it using the selected model, and output the result in the specified format.
    
    Parameters:
        audio (File): The audio file to be transcribed.
        model (str): The name of the Whisper model to use for transcription.
        output_format (str): The format of the output (e.g., "text", "json").
    
    Returns:
        str: The transcribed text in the specified format.
    """
        audio = request.files["audio"]
        model = request.form.get("model")
        output_format = request.form.get("output_format")
    
        # Initialize the Whisper client with the selected parameters
        params = get_whisper_model_params(whisper_models[model])
        whisper_client = AudioClient(inputs="audio", outputs=output_format, params=WhisperModel(**params))
    
        # Load the audio file and convert it to floating-point format
        with io.BytesIO(open(path.join("uploaded_files", "audio.wav")).read()) as audio_file:
            audio_np = np.array(BytesIO(open(audio_file)).get_array_of_samples(), dtype=np.float32) / 32767.0
    
        # Transcribe the entire audio and retrieve the transcribed result
        result = whisper_client.transcribe(audio_np, task="transcribe")
    
        # Specify the options
        options = {
            "max_line_width": 60,   # Set the max line width as needed
            "max_line_count": 10,   # Set the max line count as needed
            "highlight_words": True  # Set to True or False as needed
        }
    
        # Extract segment information from the result
        segments = result.segments
    
        # Format timestamp information for each segment
        timestamps = {
            "start": None,  # initialize start time as None
            "end": None,  # initialize end time as None
            "duration": None  # initialize duration as None
        }
        for segment in segments:
            segment_text = segment["alternatives"][0]["transcript"].replace("\n", " ")  # remove newline characters from the text
            if timestamps["start"] is None:
                timestamps["start"] = int(segment["start"] / 1000)  # convert start time to seconds
                timestamps["end"] = int(segment["start"] / 1000)  # set end time to the start time of the segment
            elif int(segment["start"] / 1000) > int(timestamps["start"] / 1000):
                timestamps["end"] = int(segment["start"] / 1000)  # set end time to the start time of the segment
            else:
                timestamps["end"] = int((segment["start"] + segment_text.index(" ")) / 1000)  # calculate end time based on the last space in the text
            duration = int(timestamps["end"] - timestamps["start"])  # calculate duration in seconds
            segments[segment]["timestamp"] = format_timestamp(timestamps)
    
        # Initialize the speech recognition client
        with gr.Interface(f"{whisper_models[model]['params']}.SpeechClient", inputs="audio", outputs=output_format) as interface:
            with io.BytesIO() as audio_file:
                audio.save(audio_file, format="wav")
                audio_file.seek(0)
                
                # Transcribe the audio file
                result = interface.run_async(audio_file)
                
                # Get the transcription result from the model
                transcription = result.get()["results"][0]["alternatives"][0]["transcript"]
            
        return render_template("result.html", transcription=transcription, output_format=output_format)

if __name__ == "__main__":
    app.run(port=5000, debug=True)