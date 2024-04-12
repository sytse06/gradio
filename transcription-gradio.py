import gradio as gr
import numpy as np
import whisper
#from whisper.utils import get_writer  # Ensure this points to your custom get_writer implementation
from custom_whisper_utils import get_writer
import os
import pydub
from pydub import AudioSegment
import tempfile

# Ensure the ResultWriter subclasses (WriteTXT, WriteVTT, etc.) are correctly defined here

import gradio as gr
import numpy as np
import whisper
#from whisper.utils import get_writer  # Ensure this points to your custom get_writer implementation
from custom_whisper_utils import get_writer
import os
import pydub
from pydub import AudioSegment
import tempfile

# Ensure the ResultWriter subclasses (WriteTXT, WriteVTT, etc.) are correctly defined here

def processAudio(audio1, audio2, model_choice, output_format):
    # Load the Whisper model based on the user's selection from the dropdown
    model = whisper.load_model(model_choice)

    # Decide which audio file to process
    audio_file_path = audio1 if audio1 is not None else audio2

    if audio_file_path is None:
        return None

    # Load and preprocess the audio file
    audio = AudioSegment.from_file(audio_file_path).set_channels(1).set_frame_rate(16000)
    result_data = {"segments": []}  # Prepare a result structure for the writer

    # Define the chunk length in milliseconds (e.g., 5 minutes)
    chunk_length_ms = 10 * 60 * 1000

    # Process the audio in chunks
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i+chunk_length_ms]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_chunk_file:
            chunk_file_path = temp_chunk_file.name
            chunk.export(chunk_file_path, format="wav")
            
            # Transcribe the chunk
            result = model.transcribe(chunk_file_path)
            result_data["segments"].append({"text": result["text"], "start": i, "end": i+chunk_length_ms})  # Example of structuring results
            
            # No need to manually clean up the temporary chunk file; it's done automatically

    # Instead of writing to a static directory, create a temporary file for the output
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}") as temp_output_file:
        output_file_path = temp_output_file.name

    # Get a writer based on the selected output format and write to the temporary file
    writer = get_writer(output_format, temp_output_file.name, is_temp=True)
    writer(result_data, output_file_path)  # Call the writer to save the result

    return output_file_path  # Return the path to the generated file for Gradio to use

iface = gr.Interface(
    fn=processAudio, 
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="Record Audio", show_label=True),
        gr.File(
            file_types=['.m4a', '.mp3', '.aac', '.wav', '.ogg', '.mp4', '.mov', '.avi', '.wmv', '.mkv', '.webm'],
            type="filepath",
            label="Upload Audio or Video",
            show_label=True,
            interactive=True,
            file_count="single"
        ),
        gr.Dropdown(label="Choose Whisper model", choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"], value="large"),
        gr.Dropdown(label="Choose output format", choices=["txt", "json", "vtt", "srt", "tsv", "all"], value="txt"),
    ],
    outputs=gr.File(label="Download transcription"),
    title="Whisper-based transcription app",
    description="Record your speech via microphone or upload an audio file and press the Submit button to transcribe it into text or other formats. Choose your preferred output format for the transcription."
)

iface.launch()