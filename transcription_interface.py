import gradio as gr
import numpy as np
import whisper
import pydub
from pydub import AudioSegment

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
    fn=processAudio2, 
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="Record Audio", show_label=True),
        gr.Audio(sources=["upload"], type="filepath", label="Upload Audio", show_label=True),
        gr.Dropdown(label="Choose Whisper model", choices=["tiny", "base", "small", "medium", "large", "large-v2"], value="base"),
    ],
    outputs=gr.Textbox(label="Transcription", lines=10, placeholder="Transcription will appear here...",show_copy_button=True),
    title="Whisper-based transcription app",
    description="Record your speech via microphone or upload an audio file and press the Submit button to transcribe it into text. Please, note that the size of the audio file should be less than 25 MB.",
    )

iface.launch()