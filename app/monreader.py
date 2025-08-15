import os
import re
import torch
import torchaudio as ta
from PIL import Image
from tempfile import NamedTemporaryFile
from flask import Flask, request, render_template, send_file
from dotenv import load_dotenv
from chatterbox import ChatterboxTTS
import google.generativeai as genai

print("Importing modules...")

# --- Load API Key ---
print("Loading env file...")
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise EnvironmentError("ERROR: GOOGLE_API_KEY not found in environment variables or .env file.")

print("Configuring Google API key...")
genai.configure(api_key=api_key)

# --- Verify API Key ---
try:
    test_model = genai.GenerativeModel('gemini-2.5-pro')
    test_response = test_model.generate_content("Warm-up check. Reply with 'OK'.")
    if "OK" in (test_response.text or "").upper():
        print("✅ Gemini API key accepted and responding.")
    else:
        print("⚠ Gemini API key responded, but did not return expected warm-up response.")
except Exception as e:
    raise ConnectionError(f"❌ Gemini API key validation failed: {e}")



# --- Flask App ---
app = Flask(__name__)

DEFAULT_PROMPT = "Extract the text from this book page. If there are two pages in the image with text, combine the results into one txt file."

# --- Text Cleaning ---
MAX_TTS_CHAR_LEN = 5000  # Pre-chunk limit

def clean_text_for_tts(text):
    text = ''.join(c for c in text if c.isprintable())
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:MAX_TTS_CHAR_LEN]

# --- Text Chunking ---
def split_text(text, max_len=300):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) <= max_len:
            current += " " + sentence
        else:
            if current:
                chunks.append(current.strip())
            current = sentence

    if current:
        chunks.append(current.strip())

    return chunks

# --- Gemini Text Extraction ---
def extract_text_gemini(image_file, prompt):
    try:
        print("Loading gemini model 'gemini-2.5-pro'")
        model = genai.GenerativeModel('gemini-2.5-pro')
        image = Image.open(image_file.stream)
        print("Image loaded successfully, extracting text.")
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"Error extracting text: {e}"

# --- TTS Generation with Optional Voice Sample ---
def generate_speech(text, voice_sample_file=None):
    try:
        print("Starting TTS generation...")
        print("\n----- RAW EXTRACTED TEXT -----")
        print(text)
        print(f"[Raw Length: {len(text)} characters]")

        cleaned_text = clean_text_for_tts(text)

        print("----- CLEANED TEXT FOR TTS -----")
        print(cleaned_text)
        print(f"[Cleaned Length: {len(cleaned_text)} characters]")

        chunks = split_text(cleaned_text, max_len=300)

        audio_prompt_path = None

        # --- Load TTS Model ---
        print("Loading TTS model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts_model = ChatterboxTTS.from_pretrained(device=device)
        if voice_sample_file:
            with NamedTemporaryFile(delete=False, suffix=".wav") as temp_voice:
                temp_voice.write(voice_sample_file.read())
                audio_prompt_path = temp_voice.name
            print(f"Using voice sample: {audio_prompt_path}")
        else:
            print("No voice sample provided. Using default voice.")

        # Generate and collect all audio chunks
        wav_chunks = []
        for idx, chunk in enumerate(chunks):
            print(f"\n-- Generating chunk {idx+1}/{len(chunks)} --")
            if audio_prompt_path:
                wav = tts_model.generate(chunk, exaggeration=0.4, cfg_weight=0.5, temperature=0.4, audio_prompt_path=audio_prompt_path)
            else:
                wav = tts_model.generate(chunk, exaggeration=0.4, cfg_weight=0.5, temperature=0.4)

            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            wav_chunks.append(wav)

        full_audio = torch.cat(wav_chunks, dim=1)

        temp_audio = NamedTemporaryFile(delete=False, suffix=".wav")
        ta.save(temp_audio.name, full_audio.cpu(), tts_model.sr)
        return temp_audio.name

    except Exception as e:
        print(f"TTS generation failed: {e}")
        return None

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def upload_and_extract():
    extracted_text = None
    audio_url = None

    if request.method == 'POST':
        image_file = request.files.get('image')
        voice_file = request.files.get('voice')  # optional
        prompt = DEFAULT_PROMPT

        if image_file:
            extracted_text = extract_text_gemini(image_file, prompt)
            audio_path = generate_speech(extracted_text, voice_file)
            if audio_path:
                audio_url = '/audio/' + os.path.basename(audio_path)
                app.config['AUDIO_PATH'] = audio_path

    return render_template("index.html", extracted_text=extracted_text, audio_url=audio_url)

@app.route('/audio/<filename>')
def serve_audio(filename):
    audio_path = os.path.join(os.path.dirname(app.config.get('AUDIO_PATH')), filename)
    return send_file(audio_path, mimetype='audio/wav')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
