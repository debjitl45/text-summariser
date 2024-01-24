from transformers import pipeline
import fitz 
import streamlit as st

summarizer = pipeline("summarization", model="Falconsai/text_summarization")

st.set_page_config(page_title="Text Summariser",
                    page_icon="🧾",
                    layout="centered",
                    initial_sidebar_state="collapsed")

st.header("Text Summariser 🧾")

input_doc=st.file_uploader("Upload a pdf file",type=["pdf"])

with fitz.open(input_doc) as doc:
    text = ""
    for page in doc:
        text += page.get_text()

submit=st.button("Summarize")
if submit:
    output=summarizer(text, max_length=50, min_length=10, do_sample=False)
    st.write(output[0]["summary_text"])
