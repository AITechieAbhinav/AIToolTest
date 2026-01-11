import streamlit as st
from transformers import pipeline
import soundfile as sf
import tempfile
import os

st.set_page_config(page_title="Text to Speech", layout="centered")
st.title("🔊 Text to Speech (Facebook MMS)")

# Load model once
tts = pipeline("text-to-speech",model="facebook/mms-tts-eng")

# User input
text = st.text_area(
    "Enter text to convert to speech",
    "Hello! This text is converted to speech using Facebook MMS model."
)

if st.button("Generate Speech"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating speech..."):
            output = tts(text)

            # Save audio to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(
                    tmp.name,
                    output["audio"],
                    output["sampling_rate"]
                )
                audio_path = tmp.name

        st.success("Speech generated successfully!")
        st.audio(audio_path)
