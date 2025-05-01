import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, BertForSequenceClassification
import torch
import spacy
import uuid
import os

# Get the absolute path to the models directory
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, "..", "..", "models")

# Load spaCy for sentence splitting
nlp = spacy.load("en_core_web_sm")

# Load NER Model
ner_model_path = os.path.join(models_dir, "ner_bert", "final")
ner_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_path)
ner_model.eval()

# Load RE Model
re_model_path = os.path.join(models_dir, "re_model", "saved_model")
re_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
re_model = BertForSequenceClassification.from_pretrained(re_model_path)
re_model.eval()

# Label mappings (example)
ner_labels = ["O", "B-THREAT_ACTOR", "I-THREAT_ACTOR", "B-MALWARE", "I-MALWARE", "B-TOOL", "I-TOOL", "B-VULNERABILITY", "B-TECHNIQUE", "B-TARGET", "B-INFRASTRUCTURE"]
re_labels = ["no_relation", "uses", "exploits", "targets", "hosts", "facilitates", "executes"]

# ========== Helper: Run NER ==========
def run_ner(text):
    inputs = ner_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = ner_model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
    tokens = ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    entities = []
    start = None
    label = None
    for i, (token, pred) in enumerate(zip(tokens, predictions)):
        tag = ner_labels[pred]
        if tag.startswith("B-"):
            if start is not None:
                entities.append({
                    "start": start,
                    "end": i,
                    "label": label,
                    "text": ner_tokenizer.convert_tokens_to_string(tokens[start:i])
                })
            start = i
            label = tag[2:]
        elif tag.startswith("I-") and label:
            continue
        else:
            if start is not None:
                entities.append({
                    "start": start,
                    "end": i,
                    "label": label,
                    "text": ner_tokenizer.convert_tokens_to_string(tokens[start:i])
                })
                start = None
                label = None
    return entities

# ========== Helper: Run RE ==========
def run_re(text, entities):
    triples = []
    for i, h in enumerate(entities):
        for j, t in enumerate(entities):
            if i == j: continue
            sent = f"[CLS] <e1> {h['text']} </e1> ... <e2> {t['text']} </e2> [SEP]"
            inputs = re_tokenizer(sent, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
            with torch.no_grad():
                output = re_model(**inputs)
            pred = torch.argmax(output.logits, dim=1).item()
            relation = re_labels[pred]
            if relation != "no_relation":
                triples.append((h["text"], relation, t["text"]))
    return triples

# ========== Helper: Convert to JSON-LD ==========
def to_jsonld(triples):
    context = {
        "@context": {
            "cti": "http://example.org/cti#",
            "source": "cti:source",
            "target": "cti:target",
            "predicate": "cti:predicate"
        }
    }
    graph = []
    for s, p, o in triples:
        graph.append({
            "@id": f"cti:{uuid.uuid4()}",
            "source": s,
            "predicate": p,
            "target": o
        })
    context["@graph"] = graph
    return context

# ========== MAIN ==========
def process_text(text, jsonld_path=None):
    sentences = [sent.text for sent in nlp(text).sents]
    all_triples = []

    for sentence in sentences:
        entities = run_ner(sentence)
        triples = run_re(sentence, entities)
        all_triples.extend(triples)

    if jsonld_path:
        jsonld = to_jsonld(all_triples)
        with open(jsonld_path, "w") as f:
            json.dump(jsonld, f, indent=2)
        print(f"[✓] Exported RDF-style triples to {jsonld_path}")
    return all_triples

# ========== CLI ==========
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="Raw CTI report or sentence")
    parser.add_argument("--output", default="triples.jsonld", help="Path to RDF/JSON-LD output")
    args = parser.parse_args()

    print(f"[•] Processing input text...")
    triples = process_text(args.text, jsonld_path=args.output)
    print(f"[✓] Extracted Triples:\n")
    for t in triples:
        print(f"  {t[0]} --[{t[1]}]--> {t[2]}")
