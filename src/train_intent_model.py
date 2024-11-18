import torch
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, Dataset
import pandas as pd

class IntentDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.labels = {label: i for i, label in enumerate(self.data['question'].unique())}
        self.label_map = {i: label for label, i in self.labels.items()}
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['question']
        label = self.labels.get(self.data.iloc[idx]['question'], -1)
        if label == -1:
            raise ValueError(f"Label not found for question: {text}")
        inputs = self.tokenizer(text, padding='max_length', max_length=128, truncation=True, return_tensors='pt')
        return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze(), torch.tensor(label)

def train_intent_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    df = pd.read_csv(r'C:\Users\admin\Desktop\customer-support-chatbot\data\processed\faq_data.csv')

    unique_labels = df['question'].nunique()
    print(f"Number of unique labels: {unique_labels}")

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=unique_labels)
    dataset = IntentDataset(r'C:\Users\admin\Desktop\customer-support-chatbot\data\processed\faq_data.csv', tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(3):
        model.train()
        total_loss = 0
        for input_ids, attention_mask, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

    model.save_pretrained(r'C:\Users\admin\Desktop\customer-support-chatbot\models\intent_model')
    tokenizer.save_pretrained(r'C:\Users\admin\Desktop\customer-support-chatbot\models\intent_model')
    print("Intent model saved at 'models/intent_model'")

if __name__ == "__main__":
    train_intent_model()
