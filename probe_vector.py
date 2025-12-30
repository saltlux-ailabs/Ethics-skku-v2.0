import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Model, GPT2Tokenizer, GPT2LMHeadModel, PreTrainedTokenizerFast, AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import time
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datasets import load_dataset
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run OpenAI Moderation API on detox JSON file.")
    parser.add_argument("--bs", type=int, default=128, help="Input JSON file path")
    parser.add_argument("--lr", type=float, default=1e-4, help="Input JSON file path")
    parser.add_argument("--layer", type=int, default=9, help="Input JSON file path")
    parser.add_argument("--model", type=str, default='gpt2', help="Input JSON file path")
    args = parser.parse_args()
        
    config = GPT2Model.from_pretrained(args.model).config
    config.output_hidden_states = True
    model = GPT2Model.from_pretrained(args.model, config=config)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    best_model_path = f"{args.model}_{args.layer}-single-probe_balance_{args.bs}_{args.lr}.pt"

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
        example["label"] = int(example["toxicity"] >= THRESH)
        return example

    def downsample_nontoxic_to_match_toxic(hf_ds, label_col="label", seed=42):
        """
        Keep all toxic samples (label=1),
        randomly sample non-toxic samples (label=0)
        so that #non-toxic == #toxic.
        """
        df = hf_ds.to_pandas()

        df_toxic = df[df[label_col] == 1]
        df_nontoxic = df[df[label_col] == 0]

        n_toxic = len(df_toxic)

        df_nontoxic_sampled = df_nontoxic.sample(
            n=n_toxic,
            random_state=seed
        )

        df_balanced = (
            pd.concat([df_toxic, df_nontoxic_sampled])
            .sample(frac=1, random_state=seed)  # shuffle
            .reset_index(drop=True)
        )

        return df_balanced

    ds = ds.map(binarize)
    train_ds_balanced = downsample_nontoxic_to_match_toxic(ds["train"])
    test_ds_balanced = downsample_nontoxic_to_match_toxic(ds["test"])

    train_dataset = ToxicityDataset(train_ds_balanced, tokenizer)
    test_dataset = ToxicityDataset(test_ds_balanced, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

    class ToxicityProbe(nn.Module):
        def __init__(self, hidden_size):
            super(ToxicityProbe, self).__init__()
            self.classifier = nn.Linear(hidden_size, 1)
            self.dropout = nn.Dropout(0.2)

        def forward(self, last_hidden_state):
            mean_last_hidden_state = last_hidden_state.mean(dim=1)
            dropped_output = self.dropout(mean_last_hidden_state)
            logits = self.classifier(dropped_output)
            return logits

    probe = ToxicityProbe(model.config.hidden_size)
    optimizer = optim.Adam(probe.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    def train(model, probe, dataloader, optimizer, criterion, device='cuda'):
        model.eval()  
        probe.train()
        total_loss, total_correct = 0, 0
        start_time = time.time()  
        
        for input_ids, attention_mask, labels in tqdm(dataloader, desc="Training", leave=False):

            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device).float()  # labels를 float 타입으로 변환
            
            with torch.no_grad():  
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[args.layer]  

            logits = probe(last_hidden_state)
            loss = criterion(logits, labels.view(-1, 1))  # labels 크기 조정
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_correct += ((torch.sigmoid(logits) > 0.5).float() == labels.view(-1, 1)).sum().item()
        
        epoch_time = time.time() - start_time
        
        accuracy = total_correct / len(dataloader.dataset)
        return total_loss / len(dataloader), accuracy, epoch_time

    def evaluate(model, probe, dataloader, criterion, device='cuda'):
        model.eval()
        probe.eval()
        total_loss, total_correct = 0, 0
        with torch.no_grad():
            for idx, (input_ids, attention_mask, labels) in enumerate(dataloader):
                input_ids, attention_mask, labels = input_ids.to(device),attention_mask.to(device), labels.to(device).float()
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[args.layer]
                
                logits = probe(last_hidden_state)
                loss = criterion(logits, labels.view(-1, 1))
                
                total_loss += loss.item()
                total_correct += ((torch.sigmoid(logits) > 0.5).float() == labels.view(-1, 1)).sum().item()

                if idx == 0:  # 첫 번째 배치만 출력
                    probabilities = torch.sigmoid(logits)
                    print(f"Sample Logits: {logits[:5].squeeze().cpu().numpy()}")
                    print(f"Sample Probabilities: {probabilities[:5].squeeze().cpu().numpy()}")
                    print(f"Sample Labels: {labels[:5].cpu().numpy()}")
        
        accuracy = total_correct / len(dataloader.dataset)
        return total_loss / len(dataloader), accuracy



    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    probe.to(device)

    for input_ids, attention_mask, label in train_loader:
        print("Input IDs:", input_ids[0])
        print("Attention Mask:", attention_mask[0])
        print("Label:", label[0])
        break

    best_val_acc = 0.0  
    epochs = 10
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        train_loss, train_acc, train_time = train(model, probe, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, probe, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Time: {train_time:.2f} seconds")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(probe.classifier.weight, best_model_path)  
            print(f"New best model saved with Validation Accuracy: {val_acc:.4f}")

    print(f"Training complete. Best Validation Accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
