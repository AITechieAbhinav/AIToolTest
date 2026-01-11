import torch
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="AI Tools by Abhinav Gupta", layout="centered")

st.title("📝 AI Tools by Abhinav Gupta")

input_text = st.text_input("Paste text below to Summarize")

with tab1:

	text = st.text_input("Paste text below to Summarize")

	if st.button("Submit") :

		summarizer = pipeline("summarization", model="google/pegasus-cnn_dailymail")
		sm_txt = summarizer(text)
		st.markdown(sm_txt)
