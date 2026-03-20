# YOUR TOKENIZER AND MODEL from PART A AND PART B RESPECTIVELY
# If you wish to change their code, please do so in their respective files under parta/ and partb/ directories.
import os
import torch
import math
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from partb.bpe_tokenizer import BPETokenizer
from parta.model import LanguageModel, collate_fn
from tqdm import tqdm

# You can also create additional files in this directory and import them here if needed.
# For example, the line below import a dummy function from utils.py file.
from .utils import dummy_function  # Replace with actual utility functions as needed

# You can structure your code as you see fit as long as the CLI works as specified.
# Finally, treat this as your FINAL MODEL TRAINING SCRIPT. Do not perform hyperparameter tuning here.
# You can create separate scripts for hyperparameter tuning if needed.

class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        with open(data_path, 'r') as f:
            self.data = [line.strip() for line in f if line.strip()]

        self.data = [self.tokenizer.encode(text) for text in self.data]
        self.char_lens = [len(text) for text in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]

        input_ids = text[:-1]  # All tokens except the last one
        label_ids = text[1:]   # All tokens except the first one
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),  # Mask of 1s for actual tokens  
            "char_len": self.char_lens[idx]
        }
    
def cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train(model,dataloader,optimizer,criterion,scheduler,accum_steps,device):
    model.train()
    total_loss = 0
    total_tokens = 0
    optimizer.zero_grad()
    for i, batch in enumerate(tqdm(dataloader)):
        input = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        labels = labels.clone()
        labels[attention_mask == 0] = -100  # Set padding token labels to -100 to ignore in loss calculation

        pred = model(input, attention_mask=attention_mask)
        loss = criterion(pred.view(-1, pred.size(-1)), labels.view(-1))
        num_tokens = attention_mask.sum().item()

        loss = loss / accum_steps  # Normalize loss for gradient accumulation
        loss.backward()

        if (i + 1) % accum_steps == 0: # Grad accum to increase gbs
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()  # Update learning rate
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps * num_tokens  # Scale back loss to original value
        total_tokens += num_tokens

    if (i + 1) % accum_steps != 0: # Final step for remaining gradients
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / total_tokens

@torch.no_grad()
def evaluate(model,dataloader,criterion,device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_chars = 0

    for batch in dataloader:
        input = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        labels = labels.clone()
        labels[attention_mask == 0] = -100  # Set padding token labels to -100 to ignore in loss calculation

        pred = model(input, attention_mask=attention_mask)
        loss = criterion(pred.view(-1, pred.size(-1)), labels.view(-1))

        num_tokens = attention_mask.sum().item()
        total_loss += loss.item() * num_tokens  # Scale loss by number of tokens
        total_tokens += num_tokens
        total_chars += sum(batch["char_len"]).item()

    avg_loss = total_loss / total_tokens
    bpc = total_loss / (total_chars * math.log(2))
    return avg_loss, bpc

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer_path)

    # Load datasets
    train_dataset = TextDataset(args.train_path, tokenizer)
    valid_dataset = TextDataset(args.valid_path, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,collate_fn=collate_fn, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=64, collate_fn=collate_fn, num_workers=8)

    # Initialize model
    vocab_size = tokenizer.get_vocab_size()
    config = {
        "d_model": 512,
        "n_heads": 8,
        "num_layers": 6,
        "d_head": 64,
        "d_ff": 2048,
        "dropout": 0.1,
        "vocab_size": vocab_size
    }
    model = LanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    train_losses, val_losses = [], []
    val_bpcs = []

    epochs = 10
    accum_steps = 4

    total_steps = (len(train_loader) // accum_steps) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    os.makedirs(args.output_model_path, exist_ok=True)

    for epoch in range(epochs):
        train_loss = train(model,train_loader,optimizer,criterion,scheduler,accum_steps,device)
        valid_loss, valid_bpc = evaluate(model,valid_loader,criterion,device)

        train_losses.append(train_loss)
        val_losses.append(valid_loss)
        val_bpcs.append(valid_bpc)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {valid_loss:.4f}, Val BPC: {valid_bpc:.4f}')

        # Save checkpoint
        torch.save(model.state_dict(), f"{args.output_model_path}/model.pth")

    # Plot loss
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_plot.png')

    # Plot Bits per Character (BPC)
    plt.figure()
    plt.plot(val_bpcs, label='Valid BPC')
    plt.xlabel('Epoch')
    plt.ylabel('BPC')
    plt.title('Validation BPC')
    plt.legend()
    plt.savefig('bpc.png')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a model on the given dataset.')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the train dataset')
    parser.add_argument('--valid_path', type=str, required=True, help='Path to the valid dataset')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer')
    parser.add_argument('--output_model_path', type=str, default='checkpoints', help='Directory to save checkpoints')

    args = parser.parse_args()
    main(args)
    # Try AMP later
