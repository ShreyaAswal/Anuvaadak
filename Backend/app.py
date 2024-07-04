import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransTokenizer import IndicProcessor
import torch
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, static_folder='../css', template_folder='../html')

# Define function to determine the model based on source and target languages
def load_model(source_lang, target_lang):
    if source_lang == "eng_Latn":
        return "en-indic"
    elif target_lang == "eng_Latn":
        return "indic-en"
    else:
        return "indic-indic"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    print(data)
    source_lang = data['source_lang']
    target_lang = data['target_lang']
    input_sentence = data['input_sentence']

    # Determine the appropriate model based on source and target languages
    model_name = "ai4bharat/indictrans2-"+load_model(source_lang, target_lang)+"-1B"
    tokenizer=AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    ip = IndicProcessor(inference=True)

    # Preprocess input sentence
    batch = ip.preprocess_batch([input_sentence], src_lang=source_lang, tgt_lang=target_lang)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenize and encode input
    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(DEVICE)

    # Generate translation
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

    # Decode and postprocess translation
    with tokenizer.as_target_tokenizer():
        generated_tokens = tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
    translation = ip.postprocess_batch(generated_tokens, lang=target_lang)

    # Prepare response
    response = {
        "source_lang": source_lang,
        "input_sentence": input_sentence,
        "target_lang": target_lang,
        "translation": translation[0]
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)