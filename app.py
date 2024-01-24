from transformers import pipeline
import fitz 
import requests
import streamlit as st

summarizer = pipeline("summarization", model="Falconsai/text_summarization")
API_URL = "https://api-inference.huggingface.co/models/atharvamundada99/bert-large-question-answering-finetuned-legal"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

st.set_page_config(page_title="Text Summariser",
                    page_icon="🧾",
                    layout="centered",
                    initial_sidebar_state="collapsed")

st.header("Text Summariser 🧾")

input_doc=st.file_uploader("Upload a pdf file",type=["pdf"])

text = ""
with fitz.open(input_doc) as doc:
    for page in doc:
        text += page.get_text()

submit=st.button("Summarize")
if submit:
    output=summarizer(text, max_length=100, min_length=50, do_sample=False)
    st.write(output[0]["summary_text"])

input_question=st.text_input("Ask a question")
ask_question=st.button("Ask")

if ask_question:
    output = query({
        "inputs": {
            "question": input_question,
            "context": text
        },
    })
    st.write(output["answer"])

    