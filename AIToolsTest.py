from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import streamlit as st

model_name = "t5-small" 
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

st.set_page_config(page_title="AI Tools by Abhinav Gupta", layout="centered")

st.title("📝 AI Tools by Abhinav Gupta")

input_text = st.text_input("Paste text below to Summarize")

if st.button("Submit"):

    inputs = tokenizer(input_text, return_tensors="pt")
    summary_ids = model.generate(
        inputs.input_ids,
        max_length=50,
        num_beams=5,
        early_stopping=True
    )
    output_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    st.markdown(output_text)
