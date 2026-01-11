import streamlit as st
import torch
from diffusers import DiffusionPipeline 

api_key = st.secrets['api_key']

st.set_page_config(page_title="AI Image Generator", layout="centered")
st.title("🎨 AI Image Generator (Stable Diffusion)")

pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-2", dtype=torch.bfloat16, device_map="cuda")

prompt = st.text_area("Enter your prompt",value=
                      """dreamlikeart, a grungy woman with rainbow hair, travelling between dimensions,dynamic pose, happy, soft eyes and narrow chin, 
                      extreme bokeh, dainty figure,long hair straight down, torn kawaii shirt and baggy jeans
                      """)

image = pipe(prompt).images[0]

st.image(image, caption="My Photo", use_container_width=True)
