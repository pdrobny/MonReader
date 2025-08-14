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
├── data
│  ├── Extended Dataset for Potential Talents.xlsx
│  ├── talents.csv
├── pageflip
│  ├── images
│  │  ├── testing
│  │  │  ├── flip
│  │  │  ├── notflip
│  │  ├── training
│  │  │  ├── flip
│  │  │  ├── notflip
├── ocr
│  ├── EDA_Embeddings.ipynb  #Data prep, EDA, Embedding models 
│  ├── Learn_to_rank.ipynb  # Pairwise ranking using Ranknet 
│  ├── starred_candidates.ipynb  # re-ranking candidates based on previously evaluated "starred candidates"
│  ├── HF_qwen.ipynb # Using LLMs to find top 5 candidates based on search term with Hugging Face and Qwen
├── tts
│  ├── HF_Llama.ipynb # Using LLMs to find top 5 candidates based on search term with Hugging Face and Llama
│  ├── HF_gemma.ipynb # Using LLMs to find top 5 candidates based on search term with Hugging Face and Gemma
│  ├── HF_gemma_instructiontuning.ipynb # improving model performance with prompt instruction tuning
├── monreader-app
│  ├── HF_gemma_finetuning.ipynb # improving model performance with finetuning
│  ├── API_groq.ipynb # Using LLMs to find top 5 candidates based on search term with Groq API
├── README.md
```
##  Installation and Setup
Pageflip detection:
   IDE:  Google Colab
   Python Version:  3.11
   Python Packages:  pandas, numpy, matplotlib, seaborn, warnings, logging, requests, sys, regex, plotly, fasttest sklearn, wordcloud, sentence_transformers, gensim, tensorflow, torch, torchview, transformers, huggingface_hub, random, peft, trl, groq
OCR text Extraction:
   IDE:  Google Colab
   Python Version:  3.11
   Python Packages:  pandas, numpy, matplotlib, seaborn, warnings, logging, requests, sys, regex, plotly, fasttest sklearn, wordcloud, sentence_transformers, gensim, tensorflow, torch, torchview, transformers, huggingface_hub, random, peft, trl, groq
Text to Speech TTS:
   IDE:  Google Colab
   Python Version:  3.11
   Python Packages:  pandas, numpy, matplotlib, seaborn, warnings, logging, requests, sys, regex, plotly, fasttest sklearn, wordcloud, sentence_transformers, gensim, tensorflow, torch, torchview, transformers, huggingface_hub, random, peft, trl, groq
App Building and Deployment:
   IDE: VS
   Web Framework: Flask
   Containerization: Docker
   Web hosting:  AWS EC2

## Data Description
The data comes from the clients sourcing efforts. Any field that could directly reveal personal details was removed and each candidate was given a unique identifier. 

Attributes:

- id: unique identifier for candidate (numeric)

- title: job title for candidate (text)

- location: geographical location for candidate (text)

- screening_score: pre-determined score 0-100 (numeric)

## Methods  
Data Collection and Preparation:  The data was checked for validity and completeness.  Multiple entries had missing or invalid titles and were removed as they would not useful for this analysis.

Expoloratory Data Analysis: A wordcloud showed data scientist, data analyst, and machine learning were frequently used words in the titles of candidates.  A box plot was used show screening score ranges by country.  Chloropeth maps were used to average screening by country.

Modeling: 
-  Embedding Models: Word embedding models (tfidf, Word2Vec, GloVe, fasttext) were used to compare all candidates to a target candidate by calculating cosine similarity. This was also done using a sentence embedding model, Sentence BERT.
-  Learn to Rank model:  A learn to rank model was created with PyTorch, RankNet and SentenceBert to compare all candidates to a target candidate. The output is a list of probabilities a candidate would rank lower than a target candidate.  Top candidates having a .50 probabilty of ranking lower and lower candidates approaching 1.0 probability of ranking lower.
-  Starred Candidate ranking: Ranking system that uses SentenceBERT to compare candidates to a user inputed job title and optionally up to five user inputted or "starred" candidates.  If no starred candidates are selected the system will only compare to the inputted job title.
-  LLM(local model)Prompt Engineering:
-  LLMs (API) Prompt Engineering:  
## Conclusions
Both word embedding and sentence embedding models can calculate cosine similarity to rank candidates. However, sentence embedding models—such as SentenceBERT—are more powerful because they capture the context and meaning of an entire sentence or, in this case, a job title. In contrast, word embedding models treat individual words as vectors. To compare job titles using word embeddings, additional code is required to compute the average vector representation for each title. For the goals of this project, SentenceBERT offers a more effective solution than word embedding models, as it provides richer semantic understanding with less manual effort. 

A Learn to Rank model was developed using PyTorch and RankNet, with SentenceBERT serving as the underlying embedding model. By training on a list of candidates paired with relevance scores, the model learns to predict the probability of one candidate ranking higher or lower than a given target candidate. Results showed that stronger candidates had approximately a 50% chance of ranking lower than the target, while weaker candidates approached a near 100% likelihood of being ranked below.  While RankNet performs well when comparing dissimilar candidates, it struggles to differentiate between closely matched ones—often resulting in clusters of candidates with indistinguishable rankings. Although it may not excel at surfacing a few top contenders, RankNet proves useful for filtering out clearly unqualified candidates.

A starred candidate ranking system utilizing SentenceBERT was used to developed.  Taking user input for job title and optionally the IDs of preferred or "starred" candidates generates a list of top 5 candidates.  If no candidate IDs are entered the model will simply compare candidates to the inputed job title. If 1 to 5 candidates are selected, it will take the average embeddings of the starred candidates and calculate cosine similarity to each candidate.  

Large Language Models (LLMs) can deliver powerful results with significantly less coding, thanks to their ability to interpret natural language prompts and generate text accordingly. However, the quality of the output largely depends on the clarity and completeness of the input prompt. The more specific and detailed your instructions are—both in terms of the task and the desired format—the more likely you are to achieve the intended result.  If privacy and model customization are top priorities, LLMs can be downloaded and run locally. This approach offers greater control but demands substantial data resources and computing power. On the other hand, if minimizing hardware costs and maximizing speed are more important, accessing LLMs via API is a more efficient and scalable alternative.

