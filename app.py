import streamlit as st
import whisper
import requests
import os
import json
import subprocess
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from transformers import pipeline
from fpdf import FPDF
import re
import spacy
import openai
import os


# ✅ Download necessary NLTK models
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# ✅ Trello API Credentials
TRELLO_API_KEY = "dbda6d00d6a395c653079850c84faffc"
TRELLO_TOKEN = "ATTA2c493c7555957c31a326ea577cc5ce696773a3373a9b7cf713b108a16686a34aAD80F3B9"
TRELLO_LIST_ID = "67ca56ac8f11786b7320aa10"  # "To Do" List ID

# ✅ Load Whisper Model
st.title("🎤 AI Meeting Assistant")
st.write("Upload an audio file to transcribe, summarize, and extract action items!")

whisper_model = whisper.load_model("small")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# ----------------------------------------------
# 🔄 Convert M4A to WAV
# ----------------------------------------------
def convert_audio(input_file, output_file="converted_meeting.wav"):
    """ Converts an M4A file to WAV format using FFmpeg. """
    if os.path.exists(output_file):
        os.remove(output_file)
    command = ["ffmpeg", "-i", input_file, "-acodec", "pcm_s16le", "-ar", "16000", output_file]
    subprocess.run(command, check=True)
    return output_file

# ----------------------------------------------
# 🎤 Transcribe Audio
# ----------------------------------------------
def transcribe_audio(audio_path):
    """ Transcribes the given audio file using Whisper. """
    st.write("🎤 Transcribing Audio...")
    result = whisper_model.transcribe(audio_path)
    return result["text"]

# ----------------------------------------------
# 📝 Summarize Text
# ----------------------------------------------
def summarize_text(text):
    """ Summarizes the transcribed text. """
    st.write("📝 Summarizing Text...")
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# ----------------------------------------------
# 📌 Extract Action Items
# ----------------------------------------------
# ✅ Load API Key Securely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def extract_action_items(text):
    """ Extracts action items using OpenAI GPT API with JSON-structured output. """
    st.write("📌 Extracting Action Items with AI...")

    prompt = f"""
    Extract clear action items from the following meeting transcript.
    
    **Output must be in JSON format with this structure:**
    {{
        "action_items": [
            {{"person": "John", "task": "Handle database integration"}},
            {{"person": "Sara", "task": "Design the UI"}},
            {{"person": "Mike", "task": "Write API documentation"}},
            {{"person": "David", "task": "Deploy project on AWS"}}
        ]
    }}

    **Transcript:**
    {text}

    **Now return only the JSON output (do not include any explanations or extra text).**
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are an AI assistant that extracts structured action items from meetings."},
                      {"role": "user", "content": prompt}],
            temperature=0
        )

        raw_output = response["choices"][0]["message"]["content"].strip()
        
        # ✅ Ensure we extract only valid JSON
        try:
            json_data = json.loads(raw_output)
            action_items = [f"{item['person']}: {item['task']}" for item in json_data["action_items"]]
        except json.JSONDecodeError:
            action_items = ["⚠️ GPT returned an invalid response. Try again!"]

        return action_items

    except Exception as e:
        return [f"❌ Error: {str(e)}"]
# ----------------------------------------------
# 📄 Save Summary as PDF
# ----------------------------------------------
def save_summary_as_pdf(summary, filename="meeting_summary.pdf"):
    """ Saves the summary as a PDF file. """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, summary)
    pdf.output(filename)
    return filename

# ----------------------------------------------
# 📌 Send Action Items to Trello
# ----------------------------------------------
def send_to_trello(action_items):
    """ Sends extracted action items to Trello as task cards. """
    st.write("📌 Sending tasks to Trello...")

    for item in action_items:
        url = f"https://api.trello.com/1/cards?key={TRELLO_API_KEY}&token={TRELLO_TOKEN}&idList={TRELLO_LIST_ID}&name={item}"
        response = requests.post(url)

        if response.status_code == 200:
            st.success(f"✅ Added '{item}' to Trello")
        else:
            st.error(f"❌ Error adding '{item}' to Trello: {response.text}")

# ----------------------------------------------
# 🚀 Web UI: Upload & Process Audio
# ----------------------------------------------
uploaded_file = st.file_uploader("Upload an audio file (.m4a, .mp3, .wav)", type=["m4a", "mp3", "wav"])

if uploaded_file:
    # Save uploaded file
    audio_path = f"uploaded_{uploaded_file.name}"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())
    
    st.success(f"✅ File uploaded: {uploaded_file.name}")

    # Convert if needed
    if audio_path.endswith(".m4a"):
        st.write("🔄 Converting M4A to WAV...")
        audio_path = convert_audio(audio_path)

    # Process Meeting
    transcription = transcribe_audio(audio_path)
    summary = summarize_text(transcription)
    action_items = extract_action_items(transcription)

    # Display results
    st.subheader("📝 Meeting Summary")
    st.write(summary)

    st.subheader("📌 Action Items")
    for item in action_items:
        st.write(f"- {item}")

    # Save as PDF
    pdf_file = save_summary_as_pdf(summary)
    st.download_button(label="📥 Download Summary as PDF", data=open(pdf_file, "rb"), file_name="meeting_summary.pdf", mime="application/pdf")

    # Send to Trello
    if st.button("📌 Send to Trello"):
        send_to_trello(action_items)