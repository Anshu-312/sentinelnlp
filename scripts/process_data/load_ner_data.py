import json
import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Define NER data directory
NER_DIR = Path(PROJECT_ROOT) / "data/processed/ner"

print(f"Loading NER data from: {NER_DIR}")

# Load training data
with open(NER_DIR / "train.json") as f:
    ner_train = json.load(f)
print(f"✅ Loaded {len(ner_train)} training samples")

# Load test data
with open(NER_DIR / "test.json") as f:
    ner_test = json.load(f)
print(f"✅ Loaded {len(ner_test)} test samples")

# Print sample statistics
print("\nDataset Statistics:")
print(f"  ➤ Training samples: {len(ner_train)}")
print(f"  ➤ Test samples: {len(ner_test)}")
print(f"  ➤ Total samples: {len(ner_train) + len(ner_test)}")

# Optional: Print a sample entry
if ner_train:
    print("\nSample training entry:")
    print(json.dumps(ner_train[0], indent=2)) 