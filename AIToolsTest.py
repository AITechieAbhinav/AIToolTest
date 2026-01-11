import vertexai
from vertexai.generative_models import GenerativeModel
import streamlit as st

def generate(prompt):
 vertexai.init(project="<YOUR_PROJECT_ID>", location="us-central1")
 model = GenerativeModel("gemini-1.5-flash-001")

 responses = model.generate_content(
   prompt,
   generation_config=generation_config,
   stream=False,
  )

 st.write(responses.text)

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

prompt = st.text_input("Enter prompt")
if prompt:
 with st.spinner('Processing...'):
  generate(prompt)
