import os
import sys
from pathlib import Path

# Get the project root directory (2 levels up from this script)
PROJECT_ROOT = Path(__file__).parent.parent.parent
print(f"Project root: {PROJECT_ROOT}")

# Create processed_data directory
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Ensure the current directory is in the path for import
sys.path.append(os.path.dirname(__file__))
from preprocess_data import preprocess_dataset

# List of (input, output) file pairs to process
DATASETS = [
    # MITRE ATT&CK dataset
    (
        PROJECT_ROOT / "data/data_sources/mitre_attack/processed/enterprise_attack_cleaned.json",
        PROCESSED_DATA_DIR / "mitre_attack/enterprise_attack_preprocessed.json"
    ),
    # APT Notes dataset - using the correct filename from data-master
    (
        PROJECT_ROOT / "data/data_sources/aptnotes/raw/data-master/data-master/APTnotes.json",
        PROCESSED_DATA_DIR / "aptnotes/aptnotes_preprocessed.json"
    ),
    # NVD CVE dataset
    (
        PROJECT_ROOT / "data/data_sources/nvd_cve/processed/nvd_recent_cves.json",
        PROCESSED_DATA_DIR / "nvd_cve/nvd_cves_preprocessed.json"
    ),
    # OTRF dataset
    (
        PROJECT_ROOT / "data/data_sources/otrf/processed/otrf_analytic_summary.json",
        PROCESSED_DATA_DIR / "otrf/otrf_preprocessed.json"
    ),
    # STIX dataset
    (
        PROJECT_ROOT / "data/data_sources/stix/processed/parsed_stix_objects.json",
        PROCESSED_DATA_DIR / "stix/preprocessed_stix_objects.json"
    )
]

def batch_preprocess(datasets):
    processed_count = 0
    skipped_count = 0
    error_count = 0

    for input_path, output_path in datasets:
        print(f"\nProcessing dataset:")
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        
        try:
            if not input_path.exists():
                print(f"⚠️  Input file not found")
                skipped_count += 1
                continue

            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            preprocess_dataset(str(input_path), str(output_path))
            print(f"✅ Successfully processed")
            processed_count += 1
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            error_count += 1

    print(f"\nProcessing Summary:")
    print(f"✅ Successfully processed: {processed_count}")
    print(f"⚠️  Skipped (missing files): {skipped_count}")
    print(f"❌ Errors: {error_count}")

if __name__ == "__main__":
    print("Starting batch preprocessing...")
    batch_preprocess(DATASETS)
    print("\nBatch preprocessing complete!")
