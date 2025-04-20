import torch
import streamlit as st
from transformers import pipeline

st.title("Sentiment Analyzer")
classifier = pipeline("sentiment-analysis")

user_input = st.text_area("Enter text to analyse:")
if st.button("Analyze"):
    result = classifier(user_input)[0]
    st.write(f"**Label:** {result['label']} | **Confidence:** {round(result['score'], 2)}")

