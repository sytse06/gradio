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

def processAudio(audio1, audio2, model_choice, output_format, progress=gr.Progress(track_tqdm=True)):
    progress(0, desc="Initializing transcription...")

    # Load the Whisper model based on the user's selection from the dropdown
    model = whisper.load_model(model_choice)

    # Decide which audio file to process
    audio_file_path = audio1 if audio1 is not None else audio2
    if audio_file_path is None:
        progress(0, desc="No file uploaded.")
        return None

    # Load and preprocess the audio
    audio = AudioSegment.from_file(audio_file_path).set_channels(1).set_frame_rate(16000)
    total_length = len(audio)
    result_data = {"segments": []}

    chunk_length_ms = 10 * 60 * 1000  # 10 minutes in milliseconds
    processed_length = 0
    
    progress(0.05, desc="Starting transcription...")
    segments = range(0, total_length, chunk_length_ms)  # Define the range for tqdm

    for i in progress.tqdm(segments, desc="Transcribing"):
        chunk = audio[i:i + chunk_length_ms]
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_chunk_file:
            chunk.export(temp_chunk_file.name, format="wav")
            result = model.transcribe(temp_chunk_file.name)
            result_data["segments"].append({"text": result["text"], "start": i, "end": i + chunk_length_ms})
        
        processed_length += len(chunk)

    progress(1, desc="Transcription complete. Processing file...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}") as temp_output_file:
        output_file_path = temp_output_file.name

    # Save the transcription using the appropriate writer
    writer = get_writer(output_format, output_file_path, is_temp=True)
    writer(result_data, audio_file_path)  # Pass the result and original audio path
    
    return output_file_path

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
    description="Record your speech via microphone or upload an audio file and press the Submit button to transcribe it into text or other formats. Choose your preferred output format for the transcription.",
    concurrency_limit=10  # Set concurrency limit at the interface level
)

if __name__ == "__main__":
    iface.launch(max_threads=10)  # Adjust the max_threads parameter as needed for performance optimization