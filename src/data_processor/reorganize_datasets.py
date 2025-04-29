"""
Dataset Reorganization and Cleaning for SentinelNLP

This script reorganizes and cleans the existing datasets to prepare them for use 
in the Ontology-Based NLP Framework for Cyber Threat Knowledge Representation.
"""

import os
import shutil
import argparse
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# Import the data processor component
try:
    # When imported as a module
    from src.data_processor.data_processor import DataProcessor, DataCleaner, DataSplitter
except ImportError:
    # When run as a script
    try:
        from data_processor import DataProcessor, DataCleaner, DataSplitter
    except ImportError:
        # When run from the parent directory
        from src.data_processor.data_processor import DataProcessor, DataCleaner, DataSplitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetReorganizer:
    """Class for reorganizing and cleaning the existing cybersecurity datasets."""
    
    def __init__(self, 
                 source_dir: str, 
                 output_dir: str = 'organized_data',
                 processed_dir: str = 'processed_data',
                 nlp_datasets_dir: str = 'nlp_datasets'):
        """
        Initialize the dataset reorganizer.
        
        Args:
            source_dir: Directory containing the source data.
            output_dir: Directory to save the reorganized data.
            processed_dir: Directory containing previously processed data.
            nlp_datasets_dir: Directory containing previously prepared NLP datasets.
        """
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.processed_dir = processed_dir
        self.nlp_datasets_dir = nlp_datasets_dir
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'processed'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'nlp_ready'), exist_ok=True)
        
        # Create data processor
        self.processor = DataProcessor(output_dir=os.path.join(output_dir, 'processed'))
        self.cleaner = DataCleaner()
        self.splitter = DataSplitter()
    
    def clean_previous_datasets(self):
        """Remove previous organized datasets."""
        logger.info("Cleaning previous organized datasets...")
        
        # Clear organized data directory (preserving the directory itself)
        for item in os.listdir(self.output_dir):
            item_path = os.path.join(self.output_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
        
        # Recreate the directory structure
        os.makedirs(os.path.join(self.output_dir, 'raw'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'processed'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'nlp_ready'), exist_ok=True)
        
        logger.info("Previous datasets cleaned.")
    
    def locate_datasets(self) -> Dict[str, List[str]]:
        """
        Locate all relevant datasets in the source directory.
        
        Returns:
            Dictionary mapping dataset types to lists of file paths.
        """
        logger.info(f"Locating datasets in {self.source_dir}...")
        
        datasets = {
            'attack_techniques': [],
            'vulnerabilities': [],
            'apt_reports': [],
            'mitre_data': [],
            'stix_data': [],
            'cve_data': [],
            'other': []
        }
        
        # Walk through the source directory
        for root, _, files in os.walk(self.source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_lower = file.lower()
                
                # Skip hidden files
                if file.startswith('.'):
                    continue
                
                # Categorize by filename pattern
                if 'attack' in file_lower and ('technique' in file_lower or 'tactic' in file_lower):
                    datasets['attack_techniques'].append(file_path)
                elif 'vuln' in file_lower or 'cve' in file_lower:
                    datasets['vulnerabilities'].append(file_path)
                elif 'apt' in file_lower or 'threat' in file_lower or 'actor' in file_lower:
                    datasets['apt_reports'].append(file_path)
                elif 'mitre' in file_lower or 'attack' in file_lower:
                    datasets['mitre_data'].append(file_path)
                elif 'stix' in file_lower or 'taxii' in file_lower:
                    datasets['stix_data'].append(file_path)
                elif 'cve' in file_lower or 'nvd' in file_lower:
                    datasets['cve_data'].append(file_path)
                else:
                    # Try to determine by file content (for CSV/JSON)
                    if file_lower.endswith(('.csv', '.json', '.jsonl')):
                        try:
                            # Check the first few lines or field names
                            if file_lower.endswith('.csv'):
                                df = pd.read_csv(file_path, nrows=5)
                                headers = list(df.columns)
                            else:  # JSON or JSONL
                                if file_lower.endswith('.jsonl'):
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        first_line = f.readline().strip()
                                    if first_line:
                                        import json
                                        headers = list(json.loads(first_line).keys())
                                    else:
                                        headers = []
                                else:
                                    import json
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        data = json.load(f)
                                    if isinstance(data, list) and data:
                                        headers = list(data[0].keys())
                                    elif isinstance(data, dict):
                                        headers = list(data.keys())
                                    else:
                                        headers = []
                            
                            # Categorize based on headers
                            headers_str = ' '.join(headers).lower()
                            if any(x in headers_str for x in ['attack', 'technique', 'tactic', 'procedure']):
                                datasets['attack_techniques'].append(file_path)
                            elif any(x in headers_str for x in ['vulnerability', 'cve', 'cvss']):
                                datasets['vulnerabilities'].append(file_path)
                            elif any(x in headers_str for x in ['apt', 'threat', 'actor', 'group']):
                                datasets['apt_reports'].append(file_path)
                            else:
                                datasets['other'].append(file_path)
                        except Exception as e:
                            logger.warning(f"Error examining {file_path}: {str(e)}")
                            datasets['other'].append(file_path)
                    else:
                        datasets['other'].append(file_path)
        
        # Add already processed data
        if os.path.exists(self.processed_dir):
            for file in os.listdir(self.processed_dir):
                file_path = os.path.join(self.processed_dir, file)
                file_lower = file.lower()
                
                if file_lower.endswith('.csv') or file_lower.endswith('.json'):
                    if 'attack_technique' in file_lower:
                        datasets['attack_techniques'].append(file_path)
                    elif 'vulnerabilit' in file_lower:
                        datasets['vulnerabilities'].append(file_path)
                    elif 'apt' in file_lower or 'report' in file_lower:
                        datasets['apt_reports'].append(file_path)
        
        # Add NLP datasets
        if os.path.exists(self.nlp_datasets_dir):
            for file in os.listdir(self.nlp_datasets_dir):
                file_path = os.path.join(self.nlp_datasets_dir, file)
                file_lower = file.lower()
                
                if 'attack_pattern' in file_lower:
                    datasets['attack_techniques'].append(file_path)
                elif 'vulnerabilit' in file_lower:
                    datasets['vulnerabilities'].append(file_path)
                elif 'apt' in file_lower or 'report' in file_lower:
                    datasets['apt_reports'].append(file_path)
        
        # Log dataset counts
        for dataset_type, files in datasets.items():
            logger.info(f"Found {len(files)} {dataset_type} files")
        
        return datasets
    
    def copy_raw_data(self, datasets: Dict[str, List[str]]):
        """
        Copy raw data files to the organized structure.
        
        Args:
            datasets: Dictionary mapping dataset types to lists of file paths.
        """
        logger.info("Copying raw data files...")
        
        # Create raw data subdirectories
        for dataset_type in datasets.keys():
            os.makedirs(os.path.join(self.output_dir, 'raw', dataset_type), exist_ok=True)
        
        # Copy files
        for dataset_type, file_paths in datasets.items():
            for file_path in file_paths:
                # Skip directories and non-existent files
                if not os.path.isfile(file_path):
                    continue
                
                # Get base filename
                filename = os.path.basename(file_path)
                
                # Create destination path
                dest_path = os.path.join(self.output_dir, 'raw', dataset_type, filename)
                
                # Copy file
                try:
                    shutil.copy2(file_path, dest_path)
                    logger.debug(f"Copied {file_path} to {dest_path}")
                except Exception as e:
                    logger.error(f"Error copying {file_path}: {str(e)}")
        
        logger.info("Raw data files copied.")
    
    def process_attack_techniques(self):
        """Process and clean attack techniques data."""
        logger.info("Processing attack techniques data...")
        
        raw_dir = os.path.join(self.output_dir, 'raw', 'attack_techniques')
        processed_dir = os.path.join(self.output_dir, 'processed', 'attack_techniques')
        nlp_dir = os.path.join(self.output_dir, 'nlp_ready', 'attack_techniques')
        
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(nlp_dir, exist_ok=True)
        
        # Find CSV files
        csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
        
        if not csv_files:
            logger.warning("No CSV attack techniques files found.")
            return
        
        # Load and merge all CSV files
        dfs = []
        for csv_file in csv_files:
            file_path = os.path.join(raw_dir, csv_file)
            try:
                df = pd.read_csv(file_path)
                # Clean dataframe
                df = self.cleaner.clean_dataframe(df)
                dfs.append(df)
                logger.info(f"Processed {csv_file} with {len(df)} rows")
            except Exception as e:
                logger.error(f"Error processing {csv_file}: {str(e)}")
        
        if not dfs:
            logger.warning("No valid attack techniques data found.")
            return
        
        # Merge dataframes
        merged_df = self.processor.merge_datasets(dfs)
        
        # Save processed data
        merged_df.to_csv(os.path.join(processed_dir, 'attack_techniques.csv'), index=False)
        logger.info(f"Saved merged attack techniques data with {len(merged_df)} rows")
        
        # Create NLP-ready datasets
        train_df, val_df, test_df = self.splitter.split_data(merged_df)
        
        # Save splits
        for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            # Save as CSV
            df.to_csv(os.path.join(nlp_dir, f'attack_techniques_{split_name}.csv'), index=False)
            
            # Save as JSONL
            with open(os.path.join(nlp_dir, f'attack_techniques_{split_name}.jsonl'), 'w', encoding='utf-8') as f:
                for _, row in df.iterrows():
                    row_dict = row.replace({pd.NA: None}).to_dict()
                    import json
                    f.write(json.dumps(row_dict) + '\n')
        
        logger.info(f"Created NLP-ready datasets: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    def process_vulnerabilities(self):
        """Process and clean vulnerabilities data."""
        logger.info("Processing vulnerabilities data...")
        
        raw_dir = os.path.join(self.output_dir, 'raw', 'vulnerabilities')
        cve_dir = os.path.join(self.output_dir, 'raw', 'cve_data')
        processed_dir = os.path.join(self.output_dir, 'processed', 'vulnerabilities')
        nlp_dir = os.path.join(self.output_dir, 'nlp_ready', 'vulnerabilities')
        
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(nlp_dir, exist_ok=True)
        
        # Find all vulnerability files
        vuln_files = []
        for d in [raw_dir, cve_dir]:
            if os.path.exists(d):
                vuln_files.extend([os.path.join(d, f) for f in os.listdir(d) 
                                 if f.endswith(('.csv', '.json', '.jsonl'))])
        
        if not vuln_files:
            logger.warning("No vulnerability data files found.")
            return
        
        # Load and process files
        dfs = []
        for file_path in vuln_files:
            try:
                # Determine file type and load
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file_path.endswith('.json'):
                    df = pd.read_json(file_path)
                elif file_path.endswith('.jsonl'):
                    import json
                    records = []
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                records.append(json.loads(line))
                    df = pd.DataFrame(records)
                
                # Clean dataframe
                df = self.cleaner.clean_dataframe(df)
                dfs.append(df)
                logger.info(f"Processed {os.path.basename(file_path)} with {len(df)} rows")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
        
        if not dfs:
            logger.warning("No valid vulnerability data found.")
            return
        
        # Merge dataframes
        merged_df = self.processor.merge_datasets(dfs)
        
        # Save processed data
        merged_df.to_csv(os.path.join(processed_dir, 'vulnerabilities.csv'), index=False)
        logger.info(f"Saved merged vulnerability data with {len(merged_df)} rows")
        
        # Create NLP-ready datasets
        train_df, val_df, test_df = self.splitter.split_data(merged_df)
        
        # Save splits
        for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            # Save as CSV
            df.to_csv(os.path.join(nlp_dir, f'vulnerabilities_{split_name}.csv'), index=False)
            
            # Save as JSONL
            with open(os.path.join(nlp_dir, f'vulnerabilities_{split_name}.jsonl'), 'w', encoding='utf-8') as f:
                for _, row in df.iterrows():
                    row_dict = row.replace({pd.NA: None}).to_dict()
                    import json
                    f.write(json.dumps(row_dict) + '\n')
        
        logger.info(f"Created NLP-ready datasets: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    def process_apt_reports(self):
        """Process and clean APT reports data."""
        logger.info("Processing APT reports data...")
        
        raw_dir = os.path.join(self.output_dir, 'raw', 'apt_reports')
        processed_dir = os.path.join(self.output_dir, 'processed', 'apt_reports')
        nlp_dir = os.path.join(self.output_dir, 'nlp_ready', 'apt_reports')
        
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(nlp_dir, exist_ok=True)
        
        # Find all APT report files
        apt_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) 
                    if f.endswith(('.csv', '.json', '.jsonl'))]
        
        if not apt_files:
            logger.warning("No APT report files found.")
            return
        
        # Load and process files
        dfs = []
        for file_path in apt_files:
            try:
                # Determine file type and load
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file_path.endswith('.json'):
                    df = pd.read_json(file_path)
                elif file_path.endswith('.jsonl'):
                    import json
                    records = []
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                records.append(json.loads(line))
                    df = pd.DataFrame(records)
                
                # Clean dataframe
                df = self.cleaner.clean_dataframe(df)
                dfs.append(df)
                logger.info(f"Processed {os.path.basename(file_path)} with {len(df)} rows")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
        
        if not dfs:
            logger.warning("No valid APT report data found.")
            return
        
        # Merge dataframes
        merged_df = self.processor.merge_datasets(dfs)
        
        # Save processed data
        merged_df.to_csv(os.path.join(processed_dir, 'apt_reports.csv'), index=False)
        logger.info(f"Saved merged APT report data with {len(merged_df)} rows")
        
        # Create NLP-ready datasets
        train_df, val_df, test_df = self.splitter.split_data(merged_df)
        
        # Save splits
        for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            # Save as CSV
            df.to_csv(os.path.join(nlp_dir, f'apt_reports_{split_name}.csv'), index=False)
            
            # Save as JSONL
            with open(os.path.join(nlp_dir, f'apt_reports_{split_name}.jsonl'), 'w', encoding='utf-8') as f:
                for _, row in df.iterrows():
                    row_dict = row.replace({pd.NA: None}).to_dict()
                    import json
                    f.write(json.dumps(row_dict) + '\n')
        
        logger.info(f"Created NLP-ready datasets: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    def create_integrated_dataset(self):
        """Create an integrated dataset for entity relationship extraction."""
        logger.info("Creating integrated dataset for entity relationship extraction...")
        
        # Directories for processed data
        processed_dir = os.path.join(self.output_dir, 'processed')
        nlp_dir = os.path.join(self.output_dir, 'nlp_ready')
        
        # Create output directory
        os.makedirs(os.path.join(nlp_dir, 'integrated'), exist_ok=True)
        
        # Load processed datasets
        datasets = {}
        for dataset_type in ['attack_techniques', 'vulnerabilities', 'apt_reports']:
            dataset_path = os.path.join(processed_dir, dataset_type, f"{dataset_type}.csv")
            if os.path.exists(dataset_path):
                try:
                    datasets[dataset_type] = pd.read_csv(dataset_path)
                    logger.info(f"Loaded {dataset_type} with {len(datasets[dataset_type])} rows")
                except Exception as e:
                    logger.error(f"Error loading {dataset_path}: {str(e)}")
        
        if not datasets:
            logger.warning("No processed datasets found for integration.")
            return
        
        # Create integrated dataset for entity relationship extraction
        # This requires selecting and renaming columns to a standard schema
        
        integrated_records = []
        
        # Process attack techniques
        if 'attack_techniques' in datasets:
            df = datasets['attack_techniques']
            # Select key columns for text analysis
            if 'description' in df.columns and 'name' in df.columns and 'id' in df.columns:
                for _, row in df.iterrows():
                    record = {
                        'source_id': row['id'],
                        'source_type': 'attack_technique',
                        'title': row.get('name', ''),
                        'text': row.get('description', ''),
                        'additional_fields': {}
                    }
                    
                    # Add additional fields that might contain relationship information
                    for col in df.columns:
                        if col not in ['id', 'name', 'description']:
                            record['additional_fields'][col] = row.get(col, '')
                    
                    integrated_records.append(record)
        
        # Process vulnerabilities
        if 'vulnerabilities' in datasets:
            df = datasets['vulnerabilities']
            # Select key columns for text analysis
            if 'description' in df.columns and 'id' in df.columns:
                for _, row in df.iterrows():
                    record = {
                        'source_id': row['id'],
                        'source_type': 'vulnerability',
                        'title': row.get('name', row.get('title', row.get('id', ''))),
                        'text': row.get('description', ''),
                        'additional_fields': {}
                    }
                    
                    # Add additional fields that might contain relationship information
                    for col in df.columns:
                        if col not in ['id', 'name', 'title', 'description']:
                            record['additional_fields'][col] = row.get(col, '')
                    
                    integrated_records.append(record)
        
        # Process APT reports
        if 'apt_reports' in datasets:
            df = datasets['apt_reports']
            # APT reports might have different column names
            text_columns = [col for col in df.columns if 'content' in col.lower() or 'text' in col.lower() 
                           or 'description' in col.lower() or 'report' in col.lower()]
            
            title_columns = [col for col in df.columns if 'title' in col.lower() or 'name' in col.lower() 
                            or 'filename' in col.lower()]
            
            id_columns = [col for col in df.columns if 'id' in col.lower() or 'identifier' in col.lower()]
            
            if text_columns:
                text_col = text_columns[0]
                title_col = title_columns[0] if title_columns else None
                id_col = id_columns[0] if id_columns else None
                
                for _, row in df.iterrows():
                    record = {
                        'source_id': row.get(id_col, '') if id_col else f"apt_report_{_}",
                        'source_type': 'apt_report',
                        'title': row.get(title_col, '') if title_col else '',
                        'text': row.get(text_col, ''),
                        'additional_fields': {}
                    }
                    
                    # Add additional fields that might contain relationship information
                    for col in df.columns:
                        if col != text_col and (not title_col or col != title_col) and (not id_col or col != id_col):
                            record['additional_fields'][col] = row.get(col, '')
                    
                    integrated_records.append(record)
        
        if not integrated_records:
            logger.warning("No records found for integrated dataset.")
            return
        
        # Create DataFrame from integrated records
        integrated_df = pd.DataFrame(integrated_records)
        
        # Save integrated dataset
        integrated_df.to_csv(os.path.join(nlp_dir, 'integrated', 'integrated_dataset.csv'), index=False)
        
        # Save as JSONL
        with open(os.path.join(nlp_dir, 'integrated', 'integrated_dataset.jsonl'), 'w', encoding='utf-8') as f:
            for _, row in integrated_df.iterrows():
                row_dict = row.replace({pd.NA: None}).to_dict()
                import json
                f.write(json.dumps(row_dict) + '\n')
        
        logger.info(f"Created integrated dataset with {len(integrated_df)} records")
        
        # Split into train/val/test
        train_df, val_df, test_df = self.splitter.split_data(integrated_df)
        
        # Save splits
        for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            # Save as CSV
            df.to_csv(os.path.join(nlp_dir, 'integrated', f'integrated_{split_name}.csv'), index=False)
            
            # Save as JSONL
            with open(os.path.join(nlp_dir, 'integrated', f'integrated_{split_name}.jsonl'), 'w', encoding='utf-8') as f:
                for _, row in df.iterrows():
                    row_dict = row.replace({pd.NA: None}).to_dict()
                    import json
                    f.write(json.dumps(row_dict) + '\n')
        
        logger.info(f"Created integrated dataset splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    def run(self, clean_previous: bool = True):
        """
        Run the complete dataset reorganization process.
        
        Args:
            clean_previous: Whether to clean previous organized datasets.
        """
        if clean_previous:
            self.clean_previous_datasets()
        
        # Locate datasets
        datasets = self.locate_datasets()
        
        # Copy raw data
        self.copy_raw_data(datasets)
        
        # Process each type of data
        self.process_attack_techniques()
        self.process_vulnerabilities()
        self.process_apt_reports()
        
        # Create integrated dataset
        self.create_integrated_dataset()
        
        logger.info("Dataset reorganization complete.")

def main():
    parser = argparse.ArgumentParser(description='Reorganize cybersecurity datasets for SentinelNLP.')
    parser.add_argument('--source-dir', type=str, default='data', help='Directory containing the source data')
    parser.add_argument('--output-dir', type=str, default='organized_data', help='Directory to save the reorganized data')
    parser.add_argument('--processed-dir', type=str, default='processed_data', help='Directory containing previously processed data')
    parser.add_argument('--nlp-datasets-dir', type=str, default='nlp_datasets', help='Directory containing previously prepared NLP datasets')
    parser.add_argument('--no-clean', action='store_true', help='Do not clean previous organized datasets')
    
    args = parser.parse_args()
    
    reorganizer = DatasetReorganizer(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        processed_dir=args.processed_dir,
        nlp_datasets_dir=args.nlp_datasets_dir
    )
    
    reorganizer.run(clean_previous=not args.no_clean)

if __name__ == "__main__":
    main() 