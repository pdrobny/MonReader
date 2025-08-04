import os
from flask import Flask, request, render_template_string, send_file
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
from chatterbox import ChatterboxTTS  # Update to your actual import
from chatterbox.vc import ChatterboxVC
import torch
import torchaudio as ta
#from scipy.io.wavfile import write as write_wav
import numpy as np

# Load API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-pro')

# Initialize Flask app
app = Flask(__name__)

# Initialize Chatterbox model
device = "cuda" if torch.cuda.is_available() else "cpu"
tts_model = ChatterboxTTS.from_pretrained(device=device)

DEFAULT_PROMPT = "Extract the text from this book page. If there are two pages in the image with text, combine the results into one txt file."

# ---- Gemini Text Extraction ----
def extract_text_gemini(image_file, prompt):
    try:
        image = Image.open(image_file.stream)
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"Error extracting text: {e}"

# ---- TTS Generation with Optional Voice Sample ----
def generate_speech(text, voice_sample_file=None):
    try:
        # Save uploaded voice sample to a temporary file if provided
        audio_prompt_path = None
        if voice_sample_file:
            with NamedTemporaryFile(delete=False, suffix=".wav") as temp_voice:
                temp_voice.write(voice_sample_file.read())
                audio_prompt_path = temp_voice.name

        # Use voice cloning if audio_prompt_path is available
        if audio_prompt_path:
            wav = tts_model.generate(text, audio_prompt_path=audio_prompt_path)
        else:
            wav = tts_model.generate(text)

       

        # Save to temporary WAV file
        temp_audio = NamedTemporaryFile(delete=False, suffix=".wav")
        ta.save(temp_audio.name, wav, tts_model.sr)  # Adjust sample rate if needed
        return temp_audio.name

    except Exception as e:
        print(f"TTS generation failed: {e}")
        return None

# ---- HTML Template ----
HTML_FORM = """
<!doctype html>
<html lang="en">
  <head>
    <title>PDTTS</title>
    <style>
      body { font-family: sans-serif; max-width: 700px; margin: auto; padding: 20px; }
      textarea { width: 100%; height: 300px; }
    </style>
  </head>
  <body>
    <h2>Upload Image to Extract Text and Generate Speech</h2>
    <form method="POST" enctype="multipart/form-data">
      <label>Image of book page:</label><br>
      <input type="file" name="image" accept="image/*" required><br><br>

      <label>Optional voice sample (for cloning):</label><br>
      <input type="file" name="voice" accept="audio/*"><br><br>
  
      <button type="submit">Extract and Generate Voice</button>
    </form>

    {% if extracted_text %}
      <hr>
      <h3>Extracted Text:</h3>
      <textarea readonly>{{ extracted_text }}</textarea>

      {% if audio_url %}
        <h3>Generated Audio:</h3>
        <audio controls>
          <source src="{{ audio_url }}" type="audio/mpeg">
          Your browser does not support audio playback.
        </audio>
      {% endif %}
    {% endif %}
  </body>
</html>
"""

# ---- Flask Routes ----
@app.route('/', methods=['GET', 'POST'])
def upload_and_extract():
    extracted_text = None
    audio_url = None

    if request.method == 'POST':
        image_file = request.files.get('image')
        voice_file = request.files.get('voice')  # optional
        prompt = request.form.get('prompt') or DEFAULT_PROMPT

        if image_file:
            extracted_text = extract_text_gemini(image_file, prompt)
            audio_path = generate_speech(extracted_text, voice_file)
            if audio_path:
                audio_url = '/audio/' + os.path.basename(audio_path)
                app.config['AUDIO_PATH'] = audio_path

    return render_template_string(HTML_FORM, extracted_text=extracted_text, audio_url=audio_url)

@app.route('/audio/<filename>')
def serve_audio(filename):
    audio_path = os.path.join(os.path.dirname(app.config.get('AUDIO_PATH')), filename)
    return send_file(audio_path, mimetype='audio/mpeg')

if __name__ == '__main__':
    app.run(port=3000, debug=True)
