import json
import torch
import os
import sys
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from seqeval.metrics import classification_report
from sklearn.metrics import accuracy_score

# Print Python and package versions for debugging
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

MODEL_NAME = "bert-base-cased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Get the project root directory (two levels up from the script)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(f"Project root: {project_root}")

# First, collect all unique labels from both train and test data
def collect_labels(file_path):
    print(f"Collecting labels from {file_path}")
    with open(file_path) as f:
        raw_data = json.load(f)
    
    label_list = set()
    for text, ann in raw_data:
        for start, end, label in ann["entities"]:
            label_list.add(label)
    
    return sorted(list(label_list))

# Collect labels from both files
train_file = os.path.join(project_root, "data/processed/ner/train.json")
test_file = os.path.join(project_root, "data/processed/ner/test.json")

print(f"Checking if files exist:")
print(f"Train file exists: {os.path.exists(train_file)}")
print(f"Test file exists: {os.path.exists(test_file)}")

train_labels = collect_labels(train_file)
test_labels = collect_labels(test_file)

print(f"Found {len(train_labels)} unique labels in train data")
print(f"Found {len(test_labels)} unique labels in test data")

# Create label mapping
labels = sorted(list(set(train_labels + test_labels)))
label_map = {f"B-{l}": i + 1 for i, l in enumerate(labels)}
label_map.update({f"I-{l}": i + len(labels) + 1 for i, l in enumerate(labels)})
label_map["O"] = 0
id2label = {v: k for k, v in label_map.items()}

print(f"Label mapping: {label_map}")

# Load & preprocess your dataset
def load_json_data(file_path):
    print(f"Loading data from {file_path}")
    with open(file_path) as f:
        raw_data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    label_list = set()

    tokenized_texts = []
    labels = []

    for text, ann in raw_data:
        entity_labels = ["O"] * len(text.split())
        for start, end, label in ann["entities"]:
            token_start = len(text[:start].split())
            token_end = len(text[:end].split())
            if token_end > token_start:
                entity_labels[token_start] = f"B-{label}"
                for i in range(token_start + 1, token_end):
                    entity_labels[i] = f"I-{label}"
            label_list.add(label)

        tokenized = tokenizer(text.split(), is_split_into_words=True, truncation=True, padding="max_length", max_length=128)
        tokenized["labels"] = [label_map.get(l, 0) for l in entity_labels] + [0] * (128 - len(entity_labels))
        tokenized_texts.append(tokenized)

    print(f"Processed {len(tokenized_texts)} examples")
    return tokenized_texts, sorted(list(label_list))

# Load the data
train_data, labels_train = load_json_data(train_file)
test_data, labels_test = load_json_data(test_file)

# Convert to PyTorch datasets
class NERDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids']),
            'attention_mask': torch.tensor(item['attention_mask']),
            'labels': torch.tensor(item['labels'])
        }

train_dataset = NERDataset(train_data)
test_dataset = NERDataset(test_data)

# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16)

# Model and tokenizer
print(f"Loading tokenizer and model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, num_labels=len(label_map), id2label=id2label, label2id=label_map
)
model.to(DEVICE)

# Create output directory if it doesn't exist
output_dir = "./models/ner_bert"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# Training setup
print("Setting up training")
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 5

# Training loop
print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=2)
            
            # Convert to lists for metrics
            for p, l in zip(predictions.cpu().numpy(), labels.cpu().numpy()):
                all_preds.append([id2label[i] for i in p if i != -100])
                all_labels.append([id2label[i] for i in l if i != -100])
    
    # Calculate metrics
    report = classification_report(all_labels, all_preds)
    print(f"Evaluation Report (Epoch {epoch+1}):")
    print(report)
    
    # Save model checkpoint
    checkpoint_path = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
    model.save_pretrained(checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

# Save the final model
print("Saving final model...")
model.save_pretrained(os.path.join(output_dir, "final"))
print("Training complete!")
