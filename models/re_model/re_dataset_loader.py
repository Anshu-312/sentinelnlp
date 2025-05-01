from torch.utils.data import Dataset
from transformers import AutoTokenizer

class REDataset(Dataset):
    def __init__(self, data_path, tokenizer_name="bert-base-cased", max_len=128):
        import json
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        with open(data_path) as f:
            self.samples = json.load(f)

        self.relation2id = {ex["relation"] for ex in self.samples}
        self.relation2id = {label: i for i, label in enumerate(sorted(self.relation2id))}
        self.id2relation = {v: k for k, v in self.relation2id.items()}

        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        inputs = self.tokenizer(
            sample["input"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        label = self.relation2id[sample["relation"]]

        item = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": label
        }
        return item
