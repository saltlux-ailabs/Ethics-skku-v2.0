import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import os
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import argparse
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--reg_weight", type=float, default=0.1, help="Regularization loss weight")
parser.add_argument("--save_path", type=str, default=None, help="Path to save the model weights") # fill-in
args = parser.parse_args()

reg_loss_weight = args.reg_weight
save_path = args.save_path

batch_size = 128
num_labels = 7
lr = 5e-4
weight_decay = 0.01
warmup_ratio = 0.1
num_epochs = 20
reg_loss_weight = 0.01
save_path = None # fill-in

model_name = 'gpt2-large'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

categories = ["severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]
ds = load_dataset("google/civil_comments")

THRESH = 0.5

class ToxicityDataset(Dataset):
    def __init__(self, ds, tokenizer, max_length=128):
        self.ds = ds
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        text = self.ds["text"][idx]
        label = self.ds["label"][idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)

def binarize(example):
    label = []

    for label_i in categories:
        label.append(int(example[label_i]>=0))

    if sum(label) == 0:
        label.append(1)
    else:
        label.append(0)

    example["label"] = label
    return example

ds = ds.map(binarize)

train_dataset = ToxicityDataset(ds["train"], tokenizer)
test_dataset = ToxicityDataset(ds["test"], tokenizer)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

model = GPT2ForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels, problem_type="multi_label_classification"
)
model.config.pad_token_id = tokenizer.pad_token_id


for param in model.transformer.parameters():
    param.requires_grad = False 


optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_ratio * num_epochs * len(train_loader),
                                            num_training_steps=num_epochs * len(train_loader))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

def evaluate_model(model, dataloader):
    model.eval()
    all_labels, all_predictions = [], []
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(dataloader):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.float().to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss  
            logits = outputs.logits  
            total_loss += loss.item()
            num_batches += 1

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_predictions.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    val_loss = total_loss / num_batches

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return {
        "f1_weighted": f1_score(all_labels, all_predictions, average="weighted"),
        "f1_micro": f1_score(all_labels, all_predictions, average="micro"),
        "accuracy": accuracy_score(all_labels, all_predictions),
        "val_loss": val_loss
    }

def cosine_similarity_regularization(weight_matrix):

    num_labels, hidden_size = weight_matrix.shape
    norm_weights = weight_matrix / torch.norm(weight_matrix, dim=1, keepdim=True)  
    reg_loss = 0.0
    
    for i in range(num_labels):
        for j in range(i + 1, num_labels):
            cos_sim = torch.dot(norm_weights[i], norm_weights[j]) 
            reg_loss += torch.abs(cos_sim)  
    
    return reg_loss

best_f1_score = 0

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    total_reg_loss = 0
    total_batches = 0

    for input_ids, attention_mask, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.float().to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss  

        weight_matrix = model.score.weight  # [num_labels, hidden_size]
        reg_loss = cosine_similarity_regularization(weight_matrix)
        total_loss = loss + reg_loss_weight * reg_loss  # 가중합

        total_train_loss += loss.item()
        total_reg_loss += reg_loss.item()
        total_batches += 1

        total_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_train_loss = total_train_loss / total_batches
    avg_reg_loss = total_reg_loss / total_batches

    result = evaluate_model(model, val_loader)

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {avg_train_loss:.4f}, Reg Loss: {avg_reg_loss:.4f}")
    print(f"Val Loss: {result['val_loss']:.4f}")
    print(f"Validation Accuracy, F1 (Micro), F1 (Weighted): {result['accuracy']:.4f}, {result['f1_micro']:.4f}, {result['f1_weighted']:.4f}")

    if result["f1_weighted"] > best_f1_score:
        best_f1_score = result["f1_weighted"]
        torch.save(model.score.weight, save_path)

print("Training complete.")
print(f"Best F1 Weighted Score: {best_f1_score:.4f}")