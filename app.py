# app.py
from flask import Flask, render_template, request
from transformers import pipeline
from gradio_client import Client
from io import BytesIO
import PyPDF2
import pyperclip

app = Flask(__name__)

# Initialize the Gradio client
client = Client("https://suramya-summarization-3.hf.space/")

def extract_text_from_pdf(file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(file)
    num_pages = len(pdf_reader.pages)
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    manual_text = request.form.get('manual_text')
    file = request.files.get('file')
    summary = ""
    if manual_text:
        summary = client.predict(manual_text, api_name="/predict")
    elif file:
        text = extract_text_from_pdf(file)
        summary = client.predict(text, api_name="/predict")
    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
