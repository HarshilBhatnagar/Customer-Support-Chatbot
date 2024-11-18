from flask import Flask, request, jsonify
from transformers import BertForSequenceClassification, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import spacy
import torch
import re

from flask_cors import CORS
app = Flask(__name__)
CORS(app)


# Loading all the models and tokenizers
intent_tokenizer = BertTokenizer.from_pretrained(r'C:\Users\admin\Desktop\customer-support-chatbot\models\intent_model')
intent_model = BertForSequenceClassification.from_pretrained(r'C:\Users\admin\Desktop\customer-support-chatbot\models\intent_model')
intent_model.eval()

entity_nlp = spacy.load(r'C:\Users\admin\Desktop\customer-support-chatbot\models\entity_model')

dialogue_tokenizer = GPT2Tokenizer.from_pretrained(r'C:\Users\admin\Desktop\customer-support-chatbot\models\dialogue_model_augmented')
dialogue_model = GPT2LMHeadModel.from_pretrained(r'C:\Users\admin\Desktop\customer-support-chatbot\models\dialogue_model_augmented')
dialogue_model.eval()

# GPU CHECK
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
intent_model.to(device)
dialogue_model.to(device)

# Helper Function - Indent Detection
def predict_intent(text):
    inputs = intent_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = intent_model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=1).item()
    return predicted_label

# Helper Function - Entity Extraction
def extract_entities(text):
    doc = entity_nlp(text)
    entities = []

    order_id_pattern = re.compile(r'\b(?:order\s+)?\d{3,}\b', re.IGNORECASE)

    detected_order_ids = set()
    for match in order_id_pattern.finditer(text):
        order_id = match.group().replace("order", "").strip()
        detected_order_ids.add((order_id, "ORDER_ID"))

    for ent in doc.ents:
        if ent.label_ == "ORDER_ID" and not order_id_pattern.match(ent.text):
            continue
        detected_order_ids.add((ent.text, ent.label_))

    entities = list(detected_order_ids)
    return entities

# Helper function = Dialogue Response 
def generate_response(prompt, intent, entities):
    if intent == 186 and any(ent[1] == "ORDER_ID" for ent in entities):
        order_id = next(ent[0] for ent in entities if ent[1] == "ORDER_ID")
        return f"Your order {order_id} is currently being processed. Please check back later for updates."

    inputs = dialogue_tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = dialogue_model.generate(
            inputs['input_ids'],
            max_length=150,
            num_return_sequences=1,
            pad_token_id=dialogue_tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            temperature=0.8
        )
    response = dialogue_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_input = data.get('text', '')

    if not user_input:
        return jsonify({'error': 'No input text provided'}), 400

    intent = predict_intent(user_input)

    entities = extract_entities(user_input)

    response = generate_response(user_input, intent, entities)

    result = {
        'intent': intent,
        'entities': entities,
        'response': response
    }
    return jsonify(result)

@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'API is running'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
