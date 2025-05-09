from transformers import BertTokenizer, BertForTokenClassification
import torch
import os
import json

# Get the absolute path to the model directory
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "models", "ner_bert", "final")

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

print(f"[•] Loaded label mappings: {id2label}")

# Use the base BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForTokenClassification.from_pretrained(model_path)
print("[✓] Model loaded successfully")

def run_ner(text):
    """
    Run Named Entity Recognition on the input text.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        list: List of dictionaries containing entity information
    """
    if not text or not isinstance(text, str):
        print("[!] Invalid input text")
        return []
        
    try:
        print(f"\n[•] Processing text: {text}")
        
        # Tokenize the text
        tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        print(f"[•] Tokenized text: {tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])}")
        
        # Get model predictions
        with torch.no_grad():
            output = model(**tokens).logits
            predictions = torch.argmax(output, dim=2)[0]
            print(f"[•] Raw predictions: {[model.config.id2label[str(p.item())] for p in predictions]}")
        
        # Process predictions
        entities = []
        current_entity = None
        
        for i, pred in enumerate(predictions):
            label = model.config.id2label[str(pred.item())]
            
            # Skip padding tokens
            if tokens.token_to_chars(0, i) is None:
                continue
                
            start = tokens.token_to_chars(0, i).start
            end = tokens.token_to_chars(0, i).end
            
            # Handle entity boundaries
            if label.startswith("B-"):  # Beginning of entity
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "start": start,
                    "end": end,
                    "label": label[2:],  # Remove B- prefix
                    "text": text[start:end]
                }
                print(f"[•] Found new entity: {current_entity}")
            elif label.startswith("I-"):  # Inside entity
                if current_entity:
                    current_entity["end"] = end
                    current_entity["text"] = text[current_entity["start"]:end]
                    print(f"[•] Extended entity: {current_entity}")
            else:  # O label or other
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Add the last entity if exists
        if current_entity:
            entities.append(current_entity)
            
        # Filter out empty entities and normalize text
        entities = [e for e in entities if e["text"].strip()]
        for e in entities:
            e["text"] = e["text"].strip()
            
        print(f"[✓] Final entities: {entities}")
        return entities
        
    except Exception as e:
        print(f"[!] Error in NER processing: {str(e)}")
        return []

if __name__ == "__main__":
    # Test the model with sample texts
    test_texts = [
        "APT29 used Mimikatz to extract credentials from the target system.",
        "Turla Group deployed both Kazuar and Gazer malware families to maintain long-term access within defense contractors.",
        "The threat actor exploited CVE-2021-34527 to gain initial access to the network.",
        "APT10 used PlugX to target servers at IP 203.0.113.5 hosting government databases."
    ]
    
    print("\n[•] Testing model with sample texts:")
    for test_text in test_texts:
        print(f"\nInput: {test_text}")
        entities = run_ner(test_text)
        print("Detected entities:")
        for entity in entities:
            print(f"- {entity['text']} ({entity['label']})")
