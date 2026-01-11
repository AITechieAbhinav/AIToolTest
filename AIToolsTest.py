import streamlit as st
import torch
from diffusers import StableDiffusionPipeline 

api_key = st.secrets['api_key']

st.set_page_config(page_title="AI Image Generator", layout="centered")
st.title("🎨 AI Image Generator (Stable Diffusion)")

device = "cpu"

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float32, use_auth_token=api_key) 
pipe.to(device) 

prompt = st.text_area("Enter your prompt",value=
                      """dreamlikeart, a grungy woman with rainbow hair, travelling between dimensions,dynamic pose, happy, soft eyes and narrow chin, 
                      extreme bokeh, dainty figure,long hair straight down, torn kawaii shirt and baggy jeans
                      """)

image = pipe(prompt).images[0]

st.image(image, caption="My Photo", use_container_width=True)
