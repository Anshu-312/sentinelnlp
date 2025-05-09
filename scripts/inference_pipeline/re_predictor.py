# Assuming sentence-level RE
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os
import json

# Get the absolute path to the model directory
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "models", "re_model", "saved_model")

# Check if model exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model directory not found at {model_path}. Please make sure the model is downloaded.")

print(f"[•] Loading model from {model_path}")

# Load label mapping from model config
config_path = os.path.join(model_path, "config.json")
with open(config_path, 'r') as f:
    config = json.load(f)
    label2id = config['label2id']
    id2label = config['id2label']

# Map generic labels to meaningful relation names
relation_mapping = {
    "LABEL_0": "no_relation",
    "LABEL_1": "uses",
    "LABEL_2": "targets",
    "LABEL_3": "exploits",
    "LABEL_4": "deploys",
    "LABEL_5": "maintains",
    "LABEL_6": "belongs_to"
}

# Use the base BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(model_path)
print("[✓] Model loaded successfully")

def run_re(sentence_pairs):
    """
    Run Relation Extraction on pairs of entities.
    
    Args:
        sentence_pairs (list): List of (entity1, entity2) tuples
        
    Returns:
        list: List of (entity1, entity2, relation) tuples
    """
    if not sentence_pairs or not isinstance(sentence_pairs, list):
        return []
        
    try:
        relations = []
        for s1, s2 in sentence_pairs:
            if not isinstance(s1, str) or not isinstance(s2, str):
                continue
                
            # Format input for relation extraction
            text = f"{s1} [SEP] {s2}"
            encoded = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            
            # Get model predictions
            with torch.no_grad():
                output = model(**encoded).logits
                label_id = torch.argmax(output, dim=1).item()
                label = model.config.id2label[str(label_id)]
                relation = relation_mapping[label]
                
            # Only include if relation is meaningful
            if relation != "no_relation":
                relations.append((s1, s2, relation))
                
        return relations
        
    except Exception as e:
        print(f"Error in RE processing: {str(e)}")
        return []

if __name__ == "__main__":
    # Test the model with sample sentence pairs
    test_pairs = [
        ("APT29", "Mimikatz"),
        ("Emotet", "CVE-2021-34527"),
        ("Turla Group", "Kazuar"),
        ("Turla Group", "Gazer"),
        ("defense contractors", "long-term access")
    ]
    print("\n[•] Testing model with sample sentence pairs:")
    for s1, s2 in test_pairs:
        print(f"Input: {s1} - {s2}")
    relations = run_re(test_pairs)
    print("\n[✓] Detected relations:")
    for s1, s2, rel in relations:
        print(f"- {s1} {rel} {s2}")
