import streamlit as st
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from huggingface_hub import hf_hub_download
import soundfile as sf
import torch
import tempfile

# 1. Set page config
st.set_page_config(page_title="Text to Speech", layout="centered")
st.title("🔊 Text to Speech (Microsoft SpeechT5)")

# 2. Cache the model resources
@st.cache_resource
def load_model():
    # Load the main model
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    # Load the vocoder (this turns the data into audio)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    # Load the processor (tokenizer)
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    
    # Load a default speaker embedding (x vector) 
    # This specific file is a standard neutral female voice often used for demos
    embeddings_dataset = torch.load("cmu_us_awb_arctic-wav-arctic_a0009-speaker_embeds.pt")
    
    return model, vocoder, processor, embeddings_dataset

model, vocoder, processor, speaker_embeddings = load_model()

# User input
text = st.text_area(
    "Enter text to convert to speech",
    "Hello! This is a lighter and faster model called SpeechT5."
)

if st.button("Generate Speech"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating speech..."):
            # Prepare inputs
            inputs = processor(text=text, return_tensors="pt")
            
            # Generate speech
            # We pass the speaker_embeddings here to give it a voice
            with torch.no_grad():
                speech = model.generate_speech(
                    inputs["input_ids"], 
                    speaker_embeddings, 
                    vocoder=vocoder
                )

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                # Convert the tensor to a numpy array for soundfile
                sf.write(tmp_file.name, speech.numpy(), samplerate=16000)
                audio_path = tmp_file.name

        st.success("Speech generated successfully!")
        st.audio(audio_path)
