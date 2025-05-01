import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm
from re_dataset_loader import REDataset

def train_model(train_path, model_name="bert-base-cased", output_dir="saved_model", epochs=3, lr=2e-5, batch_size=16):
    dataset = REDataset(train_path, tokenizer_name=model_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=len(dataset.relation2id)
    )
    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")

    model.save_pretrained(output_dir)
    print(f"[✓] Model saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="bert-base-cased")
    parser.add_argument("--output", default="saved_model")
    args = parser.parse_args()

    train_model(args.data, args.model, args.output)
