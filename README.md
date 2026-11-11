
# Autocorrect Tool

An AI-powered spelling and grammar correction web application built using Python and Streamlit. It helps improve text accuracy while aiming to preserve the original meaning.

## Features

- Spelling correction
- Grammar and punctuation correction
- AI-based text correction using a T5 model
- Editable corrected text
- Word and alphabet counters
- Copy corrected text
- Simple web interface

## Technologies Used

- Python
- Streamlit
- Hugging Face Transformers
- T5 Grammar Model
- SymSpell
- PyTorch

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/Autocorrect-Tool.git
   ```

2. Open the project folder:

   ```bash
   cd Autocorrect-Tool
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:

   ```bash
   streamlit run app.py
   ```

## How It Works

The application uses SymSpell for spelling correction and a T5-based language model for grammar and punctuation correction. A conservative filtering step helps avoid unwanted changes to the original meaning.

The AI model may need to download its files the first time the application runs.

## Project Type

Artificial Intelligence Internship Project – Pinnacle Labs

## Author

Prabhudutt
