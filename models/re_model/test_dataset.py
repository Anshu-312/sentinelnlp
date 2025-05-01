from re_dataset_loader import REDataset
from torch.utils.data import DataLoader
import os

def main():
    # Get the absolute path to the dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "..", "..", "data", "processed", "re", "re_dataset.json")
    
    # Initialize the dataset
    dataset = REDataset(
        data_path=data_path,
        tokenizer_name="bert-base-cased",
        max_len=128
    )
    
    # Print dataset info
    print(f"Dataset size: {len(dataset)} examples")
    print(f"Number of relations: {len(dataset.relation2id)}")
    print("\nRelation mapping:")
    for relation, idx in dataset.relation2id.items():
        print(f"{relation}: {idx}")
    
    # Create a dataloader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Get a batch
    batch = next(iter(dataloader))
    print("\nBatch shape:")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"Labels shape: {batch['label'].shape}")

if __name__ == "__main__":
    main() 