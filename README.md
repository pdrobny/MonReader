# MonReader

## Project Overview
An AI-powered reading assistant designed to help visually impaired users by extracting text from images and reading it aloud. The application combines computer vision for page flip detection, OCR text extraction, and voice synthesis in a seamless, automated reading experience.

## Goals
- Predict if the page is being flipped using a single image.
- Extract text from the open pages.
- Read the text aloud with optional voice cloning
- Integrate feature into a single web based app.
   
## Project Structure
```MonReader
├── pageflip
│  ├── PaulNet_WnB.ipynb # custom CNN imageclassificaiton model with evaluation in weights and biases (wandb)
│  ├── TL_WnB.ipynb # base for testing of pre-trained imageclassificaiton models with evaluation in weights and biases (wandb)
│  ├── TL_WnB_FT.ipynb # finetuning pre-trained image classificaiton model with evaluation in weights and biases (wandb)
├── ocr
│  ├── book_images/ # book page images for OCR text extraction and model evaluation
│  ├── gemini_text_extract.ipynb # OCR using gemini-2.5-pro MLLM via gemini API
│  ├── LLaVa_text_extract.ipynb # OCR using llava-hf/llava-1.5-7b-hf MLLM
│  ├── ocr_text_extract.ipynb  # OCR evaluation of easyocr, pytessaract, paddleocr
├── tts
│  ├── tts_kokoro.ipynb # tts using kokoro
│  ├── tts_chatterbox.ipynb # tts using chatterbox
├── app # folder containing files needed for app dockerization and deployment
│  ├── templates
│  │  ├── index.html # web page formatting template for flask app
│  ├── Dockerfile # containerization file
│  ├── monreader.py # flask app code
│  ├── requirements.txt # list of required installs
├── README.md
```
##  Installation and Setup
### Pageflip detection:
- IDE:  Google Colab
- Python Version:  3.11
- Python Packages:  os, pandas, numpy, matplotlib, google.colab, sklearn, tensorflow, keras, warnings, ast
### OCR text Extraction:
- IDE:  Google Colab
- Python Version:  3.11
- Python Packages:  os, pandas, numpy, matplotlib, google.colab, google.generative.ai, datasets, pillow, torch, transformers, pytesseract, paddleocr, easyocr, nltk, warnings
### Text-to-speech TTS:
- IDE:  Google Colab
- Python Version:  3.11
- Python Packages: os, pandas, numpy, matplotlib, google.colab, sklearn, seaborn, random, wandb, ast, tensorflow, keras, warnings
### App Building and Deployment:
- IDE: VS
- Python Version: 3.10+
- Voice Synthesis:  Neural voice cloning model via Chatterbox
- Web Framework: Flask
- Containerization: Docker
- Web hosting:  AWS EC2

#### Running app Locally:
1. Start Flask App
```bash
python monreader.py
```

3. Access in Browser
```arduino
http://localhost:5000
```

#### Running with Docker
```bash
docker build -t monreader-app .
docker run -p 5000:5000 monreader-app
```

#### App usage:
1. Upload an image of a page.
2. Upload a voice sample for voice cloning (optional) or leave blank for default voice.
3. Extracted text is read aloud with a synthesized voice.

## Data Description
For the page flip detection the data provided were .jpg images clipped from a page flipping video and labelled as flip or notflip and split between training and test sets.  The training set contains ~1100 images eachof flip and not flip.  The test set contains ~1100 images eachof flip and not flip.
For OCR text extraction the images were personal photographs from 'The Theory of Everything" by Stephen Hawking.
For voice cloning an 11 sec recording of my own voice in .wav was used.

## Methods  
Page-Flip Detection:  
- Custom built CCN image classification model (PaulNet) and pre-trained CCN image classification models(ResNet50, EfficientNetB0, and MobileNetV2) were tested and evaluated.
- Weights and Biases (wandb.ai) was used to evaluate and log the accuracy of the different models, optimizers, pooling methods and number of epochs.

OCR Text Extraction: 
- Tested OCR modules: EasyOCR, pytesseract, and PaddleOCR by extracting text from an image with a single page and an image with two pages side by side.  Vertical and horizontal tilts were also test to simulate different camera angles that may occur in a real setup.  Evaluated performance with bleu score.
- Tested MLLM modules: llava-hf/llava-1.5-7b-hf and gemini-2.5-pro.  Evaluated performance with bleu score.
Text-to-speech (TTS): 
-  Tested TTS models kokoro and chatterbox.
## Conclusions
The custom model peformed poorly on the test consistantly predicting all images as flip.  For pre-trained EfficientNet peformed the worst predicting all images as flip with an notflip F1 score 0.  ResNet peformed marginally better with a noflip F1 score of 0.25.  MobileNet peformed the best with an notflip F1 score of 0.47.  MobileNet was then fine-tuned, but there was no peformance increase.  Future improvements to consider are:  1) Use higher resolution images for evualation, 2) Attempt to futher fine tune the model, or 3) Try heavier weight models.

For text extraction pytesseract performed the best out of the OCR models with a bleu score of 0.81 for a single page and 0.78 for two pages side by side.  It was the only OCR model that correctly read the two side by pages by reading the left page first then the right page, while EasyOCR and PaddleOCR treated the two pages as a single pages reading across both pages for each line causing a mash up sentences and bleu scores of 0.6 or less.  All 3 OCR models showed variablily in performance based on camera tilt.  For the MLLMs, Llava was only able to provide a summary of the text while Gemini was able to extract the text word for word.  Gemini acheived bleu scores of 0.93 for a single page and 0.97 for side-by-side pages will no noticable impact to camera tilt.  

Both Kokoro and chatterbox peformed well with reading the extracted text.  Kokoro is a very light model and extracting the text in a few seconds with complete accuracy.  Although it lacks voice cloning that chatterbox has, kokoro is semi-customizable with several voice options and the ability to blend multiple voice to create a new voice.  Chatterbox is a heavier model and takes about 3 minutes to extract a full page of text.  Voice cloning is made easy but only having to point to reference audio file when training the model and achieves amazingly accurate sounding results.  While both models peform well it comes down to what is needed for the app.  If speed is needed Kokoro is the best by far.  If voice cloning is desired than chatterbox would be the choice.

For building the app gemini was selected for OCR text extraction accessed by API for its speed and accuracy.  Chatterbox was selected as the TTS for the option of voice cloning.  The code for OCR and TTS was combined in intergrated in flask then conainerized with docker.  Running locally, the app peformed well with chatterbox still taking a few minutes to generate speech.  When deploying the app to AWS EC2 there were issues with the app crashing when trying to generate speech.  Through debugging this was due to memory exhaustion caused by chatterbox.  The work around was to change the EC2 instance from the low memory t3.micro to the largest free intance type m7i-flex.large.  For better web performance a better paid tier intance could be used or switch the TTS to Kokoro for faster peformance and reduced memory usage.

