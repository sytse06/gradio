import gradio as gr
import numpy as np
import whisper
import os
import pydub
from pydub import AudioSegment

def processAudio3(audio1, audio2, model_choice):
    # Load the Whisper model based on the user's selection from the dropdown
    model = whisper.load_model(model_choice)

    # Decide which audio file to process
    audio_file_path = audio1 if audio1 is not None else audio2

    if audio_file_path is None:
        return "No audio inputs were provided."

    # Load and preprocess the audio file
    audio = AudioSegment.from_file(audio_file_path).set_channels(1).set_frame_rate(16000)

    full_transcription = ""  # Initialize the transcription result

    # Define the chunk length in milliseconds (e.g., 5 minutes)
    chunk_length_ms = 10 * 60 * 1000

    # Process the audio in chunks
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i+chunk_length_ms]
        # Export chunk to a temporary file
        chunk_file_path = f"temp_chunk_{i}.wav"
        chunk.export(chunk_file_path, format="wav")
        
        # Transcribe the chunk
        result = model.transcribe(chunk_file_path)
        full_transcription += result["text"] + " "
        
        # Clean up the temporary chunk file
        os.remove(chunk_file_path)

    return full_transcription

def processAudio2(audio1, audio2, model_choice):
    # Load the Whisper model based on the user's selection from the dropdown
    model = whisper.load_model(model_choice)

    # Decide which audio file to process
    audio_file_path = audio1 if audio1 is not None else audio2

    if audio_file_path is None:
        return "No audio inputs were provided."
    
    # Process the selected audio file with PyDub
    audio_file = AudioSegment.from_file(audio_file_path)
    # PyDub does not directly provide file size, so we check the duration
    # and make a rough estimate or another approach to handle file size
    # This part of the code assumes you want to check the file's length in time,
    # not its file size in bytes. Adjust accordingly if you're checking size differently.
    if len(audio_file) > (10 * 60 * 1000):  # Check if longer than 10 minutes
        # Process the file with PyDub, for example, slicing to the first 10 minutes
        audio_file = audio_file[:10 * 60 * 1000]
        # Export to a temporary file if necessary
        audio_file.export("temp_audio.wav", format="wav")
        audio_file_path = "temp_audio.wav"
    
    result = model.transcribe(audio_file_path)
    print(result["text"])
    return result["text"]

def processAudio(audio1, audio2, model_choice):
    # Load the Whisper model based on the user's selection from the dropdown
    model = whisper.load_model(model_choice)

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
    result = model.transcribe(audioOk)
    print(result["text"])
    return result["text"]

iface = gr.Interface(
    fn=processAudio3, 
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="Record Audio", show_label=True),
        #gr.Audio(sources=["upload"], type="filepath", label="Upload Audio or Video", show_label=True),
        gr.File(
            file_types=['.m4a', '.mp3', '.aac', '.wav', '.ogg', '.mp4', '.mov', '.avi', '.wmv', '.mkv', '.webm'],
            type="filepath",
            label="Upload Audio or Video",
            show_label=True,
            interactive=True,
            file_count="single"  # Default is "single", explicitly stated here for clarity      
            ),
        gr.Dropdown(label="Choose Whisper model", choices=["tiny", "base", "small", "medium", "large", "large-v2",  "large-v3"], value="large"),
    ],
    outputs=gr.Textbox(label="Transcription", lines=14, placeholder="Transcription will appear here...",show_copy_button=True),
    title="Whisper-based transcription app",
    description="Record your speech via microphone or upload an audio file and press the Submit button to transcribe it into text.",
    )

iface.launch()