import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Function to save models and tokenizers locally
def save_model_and_tokenizer(model_name, local_dir):
    # Create directories if they don't exist
    model_dir = os.path.join(local_dir, "model")
    tokenizer_dir = os.path.join(local_dir, "tokenizer")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tokenizer_dir, exist_ok=True)

    # Load and save the model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    model.save_pretrained(model_dir)

    # Load and save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(tokenizer_dir)

# Define the models to download
models = {
    "en-indic": "ai4bharat/indictrans2-en-indic-1B",
    "indic-en": "ai4bharat/indictrans2-indic-en-1B",
    "indic-indic": "ai4bharat/indictrans2-indic-indic-1B"
}

# Base directory for saving models
base_dir = "ml_models"

# Download and save each model and tokenizer
for local_name, model_name in models.items():
    local_dir = os.path.join(base_dir, local_name)
    save_model_and_tokenizer(model_name, local_dir)

print("Models and tokenizers saved successfully.")
