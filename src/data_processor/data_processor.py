"""
Data Processing Component for SentinelNLP

This module provides classes for loading, cleaning, and preparing cybersecurity data 
for NLP processing and knowledge extraction.
"""

import os
import json
import csv
import re
import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataLoader(ABC):
    """Abstract base class for loading data from various sources."""
    
    @abstractmethod
    def load(self, source_path: str) -> pd.DataFrame:
        """
        Load data from the specified source.
        
        Args:
            source_path: Path to the data source.
            
        Returns:
            DataFrame containing the loaded data.
        """
        pass

class CSVDataLoader(DataLoader):
    """Loader for CSV data files."""
    
    def load(self, source_path: str) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            source_path: Path to the CSV file.
            
        Returns:
            DataFrame containing the CSV data.
        """
        logger.info(f"Loading CSV data from {source_path}")
        try:
            df = pd.read_csv(source_path)
            logger.info(f"Successfully loaded {len(df)} records from {source_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV data from {source_path}: {str(e)}")
            raise

class JSONDataLoader(DataLoader):
    """Loader for JSON data files."""
    
    def load(self, source_path: str) -> pd.DataFrame:
        """
        Load data from a JSON file.
        
        Args:
            source_path: Path to the JSON file.
            
        Returns:
            DataFrame containing the JSON data.
        """
        logger.info(f"Loading JSON data from {source_path}")
        try:
            if source_path.endswith('.jsonl'):
                # Handle JSONL (JSON Lines) format
                records = []
                with open(source_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():  # Skip empty lines
                            records.append(json.loads(line))
                df = pd.DataFrame(records)
            else:
                # Handle regular JSON format
                with open(source_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Handle different JSON structures
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    if 'data' in data and isinstance(data['data'], list):
                        df = pd.DataFrame(data['data'])
                    else:
                        df = pd.DataFrame([data])
                else:
                    raise ValueError(f"Unsupported JSON structure in {source_path}")
                    
            logger.info(f"Successfully loaded {len(df)} records from {source_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading JSON data from {source_path}: {str(e)}")
            raise

class STIXDataLoader(DataLoader):
    """Loader for STIX/TAXII data."""
    
    def load(self, source_path: str) -> pd.DataFrame:
        """
        Load data from a STIX/TAXII source.
        
        Args:
            source_path: Path to the STIX/TAXII data.
            
        Returns:
            DataFrame containing the STIX/TAXII data.
        """
        logger.info(f"Loading STIX data from {source_path}")
        try:
            # For STIX bundles
            with open(source_path, 'r', encoding='utf-8') as f:
                stix_data = json.load(f)
            
            # Extract STIX objects from bundle
            if 'objects' in stix_data and isinstance(stix_data['objects'], list):
                stix_objects = stix_data['objects']
            else:
                stix_objects = [stix_data]
            
            # Convert to DataFrame
            records = []
            for obj in stix_objects:
                record = {
                    'id': obj.get('id', ''),
                    'type': obj.get('type', ''),
                    'created': obj.get('created', ''),
                    'modified': obj.get('modified', ''),
                    'name': obj.get('name', ''),
                    'description': obj.get('description', '')
                }
                
                # Add type-specific fields
                if obj.get('type') == 'threat-actor':
                    record['aliases'] = obj.get('aliases', [])
                    record['roles'] = obj.get('roles', [])
                elif obj.get('type') == 'malware':
                    record['malware_types'] = obj.get('malware_types', [])
                    record['is_family'] = obj.get('is_family', False)
                elif obj.get('type') == 'attack-pattern':
                    record['kill_chain_phases'] = obj.get('kill_chain_phases', [])
                
                records.append(record)
            
            df = pd.DataFrame(records)
            logger.info(f"Successfully loaded {len(df)} STIX objects from {source_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading STIX data from {source_path}: {str(e)}")
            raise

class DataCleaner:
    """Class for cleaning and normalizing cybersecurity data."""
    
    def __init__(self):
        self.text_columns = []
        
    def detect_text_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Automatically detect columns that contain text data.
        
        Args:
            df: DataFrame to analyze.
            
        Returns:
            List of column names containing text data.
        """
        text_columns = []
        for col in df.columns:
            # Check if column has string data type
            if df[col].dtype == 'object':
                # Check first non-null value
                first_value = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if isinstance(first_value, str) and len(first_value) > 20:
                    text_columns.append(col)
        
        self.text_columns = text_columns
        return text_columns
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data.
        
        Args:
            text: Text to clean.
            
        Returns:
            Cleaned text.
        """
        if not isinstance(text, str):
            return ""
        
        # Replace HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common escape sequences
        text = text.replace('\\n', ' ').replace('\\t', ' ').replace('\\r', ' ')
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'([.!?])\1+', r'\1', text)
        
        return text.strip()
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize all text columns in a DataFrame.
        
        Args:
            df: DataFrame to clean.
            
        Returns:
            Cleaned DataFrame.
        """
        logger.info("Cleaning DataFrame...")
        
        # Detect text columns if not already set
        if not self.text_columns:
            self.detect_text_columns(df)
        
        # Create a copy of the DataFrame to avoid modifying the original
        cleaned_df = df.copy()
        
        # Clean text columns
        for col in self.text_columns:
            if col in cleaned_df.columns:
                logger.info(f"Cleaning text in column: {col}")
                cleaned_df[col] = cleaned_df[col].apply(self.clean_text)
        
        # Remove completely empty rows
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(how='all')
        dropped_rows = initial_rows - len(cleaned_df)
        if dropped_rows > 0:
            logger.info(f"Removed {dropped_rows} completely empty rows")
        
        # Fill NaN values with appropriate defaults based on column type
        for col in cleaned_df.columns:
            col_type = cleaned_df[col].dtype
            if col in self.text_columns:
                cleaned_df[col] = cleaned_df[col].fillna("")
            elif col_type == 'object':
                cleaned_df[col] = cleaned_df[col].fillna("")
            elif pd.api.types.is_numeric_dtype(col_type):
                cleaned_df[col] = cleaned_df[col].fillna(0)
            else:
                cleaned_df[col] = cleaned_df[col].fillna("")
        
        logger.info(f"DataFrame cleaning complete. Shape: {cleaned_df.shape}")
        return cleaned_df
    
    def deduplicate(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Remove duplicate rows from DataFrame.
        
        Args:
            df: DataFrame to deduplicate.
            columns: Columns to consider for duplication. If None, all columns are used.
            
        Returns:
            Deduplicated DataFrame.
        """
        initial_rows = len(df)
        if columns:
            deduped_df = df.drop_duplicates(subset=columns)
        else:
            deduped_df = df.drop_duplicates()
        
        dropped_rows = initial_rows - len(deduped_df)
        logger.info(f"Removed {dropped_rows} duplicate rows")
        
        return deduped_df
    
    def filter_by_language(self, df: pd.DataFrame, column: str, language: str = 'en') -> pd.DataFrame:
        """
        Filter DataFrame to include only rows with text in the specified language.
        
        Args:
            df: DataFrame to filter.
            column: Column containing text to check for language.
            language: Language code to filter for.
            
        Returns:
            Filtered DataFrame.
        """
        try:
            import langdetect
            
            def is_language(text):
                if not isinstance(text, str) or len(text.strip()) < 20:
                    return True  # Skip short strings
                try:
                    detected = langdetect.detect(text)
                    return detected == language
                except:
                    return True  # Keep rows with undetectable language
            
            initial_rows = len(df)
            filtered_df = df[df[column].apply(is_language)]
            dropped_rows = initial_rows - len(filtered_df)
            logger.info(f"Filtered out {dropped_rows} rows not in {language} language")
            
            return filtered_df
        except ImportError:
            logger.warning("langdetect not installed. Skipping language filtering.")
            return df

class DataSplitter:
    """Class for splitting data into training, validation, and test sets."""
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, 
                  random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame into training, validation, and test sets.
        
        Args:
            df: DataFrame to split.
            test_size: Fraction of data to use for testing.
            val_size: Fraction of data to use for validation.
            random_state: Random seed for reproducibility.
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Splitting data with test_size={test_size}, val_size={val_size}")
        
        # Calculate the effective validation size relative to the remaining data after test split
        effective_val_size = val_size / (1 - test_size)
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        
        # Second split: separate validation set from training set
        train_df, val_df = train_test_split(train_val_df, test_size=effective_val_size, random_state=random_state)
        
        logger.info(f"Split complete. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
                   output_dir: str, base_filename: str, formats: List[str] = ['csv', 'jsonl']) -> None:
        """
        Save data splits to disk in specified formats.
        
        Args:
            train_df: Training data.
            val_df: Validation data.
            test_df: Test data.
            output_dir: Directory to save files.
            base_filename: Base name for output files.
            formats: List of formats to save (csv, jsonl).
        """
        os.makedirs(output_dir, exist_ok=True)
        
        splits = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
        
        for split_name, df in splits.items():
            for fmt in formats:
                output_path = os.path.join(output_dir, f"{base_filename}_{split_name}.{fmt}")
                
                if fmt == 'csv':
                    df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
                elif fmt == 'jsonl':
                    with open(output_path, 'w', encoding='utf-8') as f:
                        for _, row in df.iterrows():
                            # Convert to dict and handle NaN values
                            row_dict = row.replace({pd.NA: None}).to_dict()
                            f.write(json.dumps(row_dict) + '\n')
                else:
                    logger.warning(f"Unsupported format: {fmt}")
                    continue
                
                logger.info(f"Saved {split_name} split to {output_path}")

class DataProcessor:
    """Main class that orchestrates data loading, cleaning, and splitting."""
    
    def __init__(self, output_dir: str = 'processed_data'):
        self.output_dir = output_dir
        self.loaders = {
            'csv': CSVDataLoader(),
            'json': JSONDataLoader(),
            'jsonl': JSONDataLoader(),
            'stix': STIXDataLoader()
        }
        self.cleaner = DataCleaner()
        self.splitter = DataSplitter()
        
        os.makedirs(output_dir, exist_ok=True)
    
    def process_file(self, file_path: str, output_name: str = None, split: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Process a single data file through the entire pipeline.
        
        Args:
            file_path: Path to the data file.
            output_name: Base name for output files. If None, derived from file_path.
            split: Whether to split the data into train/val/test sets.
            
        Returns:
            Dictionary of DataFrames with keys 'full', 'train', 'val', 'test' (if split=True).
        """
        # Determine file type and select appropriate loader
        file_ext = os.path.splitext(file_path)[1].lower()[1:]
        if file_ext not in self.loaders:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        loader = self.loaders[file_ext]
        
        # Load data
        logger.info(f"Processing file: {file_path}")
        df = loader.load(file_path)
        
        # Clean data
        df = self.cleaner.clean_dataframe(df)
        df = self.cleaner.deduplicate(df)
        
        # Generate output name if not provided
        if output_name is None:
            output_name = os.path.basename(os.path.splitext(file_path)[0])
        
        # Save full dataset
        output_path = os.path.join(self.output_dir, f"{output_name}.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
        result = {'full': df}
        
        # Split and save if requested
        if split:
            train_df, val_df, test_df = self.splitter.split_data(df)
            result.update({
                'train': train_df,
                'val': val_df,
                'test': test_df
            })
            
            # Save splits
            nlp_dataset_dir = os.path.join(self.output_dir, 'nlp_datasets')
            self.splitter.save_splits(
                train_df, val_df, test_df,
                nlp_dataset_dir, output_name
            )
        
        return result
    
    def process_directory(self, dir_path: str, pattern: str = '*.*', recursive: bool = True) -> List[Dict[str, pd.DataFrame]]:
        """
        Process all data files in a directory that match the given pattern.
        
        Args:
            dir_path: Path to the directory.
            pattern: Glob pattern for matching files.
            recursive: Whether to search directories recursively.
            
        Returns:
            List of dictionaries, each containing processed DataFrames.
        """
        logger.info(f"Processing directory: {dir_path} with pattern {pattern}")
        
        # Find all matching files
        if recursive:
            matched_files = list(Path(dir_path).rglob(pattern))
        else:
            matched_files = list(Path(dir_path).glob(pattern))
        
        logger.info(f"Found {len(matched_files)} files matching pattern")
        
        # Process each file
        results = []
        for file_path in matched_files:
            try:
                result = self.process_file(str(file_path))
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
        
        return results

    def merge_datasets(self, datasets: List[pd.DataFrame], merge_column: Optional[str] = None) -> pd.DataFrame:
        """
        Merge multiple datasets into a single DataFrame.
        
        Args:
            datasets: List of DataFrames to merge.
            merge_column: Column to use for merging (if None, simple concatenation is used).
            
        Returns:
            Merged DataFrame.
        """
        if not datasets:
            return pd.DataFrame()
        
        if merge_column is not None:
            # Merge using the specified column
            merged_df = datasets[0].copy()
            for df in datasets[1:]:
                merged_df = pd.merge(merged_df, df, on=merge_column, how='outer')
        else:
            # Simple concatenation
            merged_df = pd.concat(datasets, ignore_index=True)
        
        # Clean and deduplicate the merged result
        merged_df = self.cleaner.clean_dataframe(merged_df)
        merged_df = self.cleaner.deduplicate(merged_df)
        
        logger.info(f"Merged {len(datasets)} datasets into a single DataFrame with {len(merged_df)} rows")
        
        return merged_df

if __name__ == "__main__":
    # Example usage
    processor = DataProcessor(output_dir='processed_data')
    
    # Process a single file
    processor.process_file('data/sample_vulnerabilities.csv', 'vulnerabilities')
    
    # Process all CSV files in a directory
    processor.process_directory('data/raw', pattern='*.csv') 