import pandas as pd
import re
import spacy

def clean_text(text):
    """Clean input text by removing special characters and extra spaces."""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text))
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def preprocess_conversation_data():
    """Preprocess Conversation.csv for dialogue model training."""
    input_path = r'C:\Users\admin\Desktop\customer-support-chatbot\data\raw\Conversation.csv'
    output_path = r'C:\Users\admin\Desktop\customer-support-chatbot\data\processed\conversation_data.csv'
    
    df = pd.read_csv(input_path, encoding='utf-8')
    df['question'] = df['question'].apply(clean_text)
    df['answer'] = df['answer'].apply(clean_text)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed conversation data saved at '{output_path}'")

def preprocess_faq_data():
    """Preprocess FAQ.csv for intent recognition."""
    input_path = r'C:\Users\admin\Desktop\customer-support-chatbot\data\raw\FAQ.csv'
    output_path = r'C:\Users\admin\Desktop\customer-support-chatbot\data\processed\faq_data.csv'
    
    df = pd.read_csv(input_path, names=['question', 'answer'], encoding='ISO-8859-1')
    df['question'] = df['question'].apply(clean_text)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed FAQ data saved at '{output_path}'")

def preprocess_ner_data():
    """Preprocess NER dataset."""
    input_path = r'C:\Users\admin\Desktop\customer-support-chatbot\data\raw\ner_datasetreference.csv'
    output_path = r'C:\Users\admin\Desktop\customer-support-chatbot\data\processed\ner_data.csv'
    
    df = pd.read_csv(input_path, encoding='ISO-8859-1')
    df['Word'] = df['Word'].apply(clean_text)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed NER data saved at '{output_path}'")

if __name__ == "__main__":
    preprocess_conversation_data()
    preprocess_faq_data()
    preprocess_ner_data()
