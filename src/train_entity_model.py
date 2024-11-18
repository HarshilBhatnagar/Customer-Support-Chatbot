import spacy
from spacy.training import Example
import pandas as pd

def train_entity_model():
    nlp = spacy.blank("en")

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    ner.add_label("ORDER_ID")
    ner.add_label("PRODUCT_NAME")

    input_path = r'C:\Users\admin\Desktop\customer-support-chatbot\data\processed\ner_data.csv'
    df = pd.read_csv(input_path)

    df['Word'] = df['Word'].fillna('')

    training_data = []
    for _, row in df.iterrows():
        text = row['Word']
        text = str(text).strip()  
        entities = []
        if "order" in text.lower():
            entities.append((0, len(text), "ORDER_ID"))
        elif "product" in text.lower():
            entities.append((0, len(text), "PRODUCT_NAME"))

        if entities:
            training_example = Example.from_dict(nlp.make_doc(text), {"entities": entities})
            training_data.append(training_example)

    # Initializing the optimizer
    optimizer = nlp.initialize()

    for epoch in range(10):
        losses = {}
        nlp.update(training_data, drop=0.5, sgd=optimizer, losses=losses)
        print(f"Epoch {epoch+1}: Loss = {losses['ner']:.4f}")

    output_path = r'C:\Users\admin\Desktop\customer-support-chatbot\models\entity_model'
    nlp.to_disk(output_path)
    print(f"Entity model saved at '{output_path}'")

if __name__ == "__main__":
    train_entity_model()
