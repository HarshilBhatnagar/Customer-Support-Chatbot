import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class DialogueDataset(Dataset):
    def __init__(self, data_file, tokenizer):
        self.data = pd.read_csv(data_file)
        self.tokenizer = tokenizer
        self.dialogues = list(self.data['question'] + " " + tokenizer.eos_token + " " + self.data['answer'] + tokenizer.eos_token)

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]
        inputs = self.tokenizer(dialogue, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze()

def train_dialogue_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    model.to(device)

    data_file = r'C:\Users\admin\Desktop\customer-support-chatbot\data\processed\conversation_data.csv'
    dataset = DialogueDataset(data_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    model.train()
    for epoch in range(3):
        total_loss = 0
        for input_ids, attention_mask in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}")

    output_path = r'C:\Users\admin\Desktop\customer-support-chatbot\models\dialogue_model'
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Dialogue model saved at '{output_path}'")

if __name__ == "__main__":
    train_dialogue_model()
