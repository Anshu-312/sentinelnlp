"""
Unit tests for the DataProcessor class.
"""

import os
import tempfile
import unittest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.data_processor.data_processor import (
    DataLoader,
    CSVDataLoader,
    JSONDataLoader,
    DataCleaner,
    DataSplitter,
    DataProcessor
)

class TestDataCleaner(unittest.TestCase):
    """Tests for the DataCleaner class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.cleaner = DataCleaner()
        
        # Create sample data
        self.test_data = pd.DataFrame({
            'id': ['1', '2', '3', '4', '5'],
            'title': ['Sample Title 1', 'Sample Title 2', '<h1>Sample Title 3</h1>', 'Sample Title 4', 'Sample Title 5'],
            'description': [
                'This is a clean description.',
                'This description has <b>HTML tags</b> that should be removed.',
                'This description has\nmultiple\nlines.',
                'This description has excessive punctuation!!!!',
                None
            ],
            'severity': ['HIGH', 'MEDIUM', 'LOW', 'CRITICAL', None]
        })
    
    def test_detect_text_columns(self):
        """Test detection of text columns."""
        text_columns = self.cleaner.detect_text_columns(self.test_data)
        self.assertIn('description', text_columns)
        self.assertNotIn('id', text_columns)
    
    def test_clean_text(self):
        """Test cleaning of text data."""
        # Test HTML tag removal
        html_text = "<p>This is a <b>test</b> with <i>HTML</i> tags.</p>"
        cleaned_text = self.cleaner.clean_text(html_text)
        self.assertNotIn('<', cleaned_text)
        self.assertNotIn('>', cleaned_text)
        
        # Test whitespace normalization
        whitespace_text = "This  has   multiple    spaces."
        cleaned_text = self.cleaner.clean_text(whitespace_text)
        self.assertEqual(cleaned_text, "This has multiple spaces.")
        
        # Test newline handling
        newline_text = "This has\nmultiple\nlines."
        cleaned_text = self.cleaner.clean_text(newline_text)
        self.assertEqual(cleaned_text, "This has multiple lines.")
        
        # Test non-string input
        non_string = 12345
        cleaned_text = self.cleaner.clean_text(non_string)
        self.assertEqual(cleaned_text, "")
    
    def test_clean_dataframe(self):
        """Test cleaning of DataFrame."""
        cleaned_df = self.cleaner.clean_dataframe(self.test_data)
        
        # Check that HTML tags were removed
        self.assertNotIn('<h1>', cleaned_df['title'].iloc[2])
        self.assertNotIn('</h1>', cleaned_df['title'].iloc[2])
        
        # Check that newlines were normalized
        self.assertNotIn('\n', cleaned_df['description'].iloc[2])
        
        # Check that excessive punctuation was normalized
        self.assertEqual(cleaned_df['description'].iloc[3].count('!'), 1)
        
        # Check that NaN values were filled
        self.assertFalse(cleaned_df['description'].isna().any())
        self.assertFalse(cleaned_df['severity'].isna().any())
    
    def test_deduplicate(self):
        """Test deduplication of rows."""
        # Create data with duplicates
        data_with_duplicates = pd.concat([self.test_data, self.test_data.iloc[0:2]], ignore_index=True)
        self.assertEqual(len(data_with_duplicates), len(self.test_data) + 2)
        
        # Deduplicate
        deduped_df = self.cleaner.deduplicate(data_with_duplicates)
        self.assertEqual(len(deduped_df), len(self.test_data))

class TestDataSplitter(unittest.TestCase):
    """Tests for the DataSplitter class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.splitter = DataSplitter()
        
        # Create sample data
        self.test_data = pd.DataFrame({
            'id': range(100),
            'value': range(100)
        })
    
    def test_split_data(self):
        """Test splitting data into train/val/test sets."""
        train_df, val_df, test_df = self.splitter.split_data(
            self.test_data, test_size=0.2, val_size=0.1
        )
        
        # Check sizes
        self.assertEqual(len(train_df), 70)  # 70% of data
        self.assertEqual(len(val_df), 10)    # 10% of data
        self.assertEqual(len(test_df), 20)   # 20% of data
        
        # Check no overlap
        train_ids = set(train_df['id'])
        val_ids = set(val_df['id'])
        test_ids = set(test_df['id'])
        
        self.assertEqual(len(train_ids.intersection(val_ids)), 0)
        self.assertEqual(len(train_ids.intersection(test_ids)), 0)
        self.assertEqual(len(val_ids.intersection(test_ids)), 0)
    
    def test_save_splits(self):
        """Test saving data splits to disk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Split data
            train_df, val_df, test_df = self.splitter.split_data(self.test_data)
            
            # Save splits
            self.splitter.save_splits(
                train_df, val_df, test_df,
                temp_dir, 'test_data',
                formats=['csv', 'jsonl']
            )
            
            # Check files exist
            for split in ['train', 'val', 'test']:
                for fmt in ['csv', 'jsonl']:
                    file_path = os.path.join(temp_dir, f"test_data_{split}.{fmt}")
                    self.assertTrue(os.path.exists(file_path))
                    
                    # Verify file content
                    if fmt == 'csv':
                        df = pd.read_csv(file_path)
                        if split == 'train':
                            self.assertEqual(len(df), len(train_df))
                        elif split == 'val':
                            self.assertEqual(len(df), len(val_df))
                        elif split == 'test':
                            self.assertEqual(len(df), len(test_df))

class TestDataProcessor(unittest.TestCase):
    """Tests for the DataProcessor class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for output
        self.temp_dir = tempfile.TemporaryDirectory()
        self.processor = DataProcessor(output_dir=self.temp_dir.name)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'id': ['1', '2', '3'],
            'title': ['Sample Title 1', 'Sample Title 2', 'Sample Title 3'],
            'description': ['Description 1', 'Description 2', 'Description 3']
        })
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    @patch('pandas.DataFrame.to_csv')
    @patch('src.data_processor.data_processor.CSVDataLoader.load')
    def test_process_file(self, mock_load, mock_to_csv):
        """Test processing a single file."""
        # Mock the loader to return sample data
        mock_load.return_value = self.sample_data
        
        # Call the method
        result = self.processor.process_file('fake_file.csv', 'test_output', split=False)
        
        # Check that the loader was called
        mock_load.assert_called_once_with('fake_file.csv')
        
        # Check that to_csv was called
        mock_to_csv.assert_called()
        
        # Check the result
        self.assertIn('full', result)
        self.assertEqual(len(result['full']), len(self.sample_data))
    
    @patch('src.data_processor.data_processor.DataProcessor.process_file')
    def test_process_directory(self, mock_process_file):
        """Test processing a directory of files."""
        # Create temporary test files
        with tempfile.TemporaryDirectory() as test_dir:
            # Create test CSV files
            for i in range(3):
                file_path = os.path.join(test_dir, f"test_{i}.csv")
                with open(file_path, 'w') as f:
                    f.write("id,title,description\n")
                    f.write(f"{i},Title {i},Description {i}\n")
            
            # Mock process_file to return a dummy result
            mock_process_file.return_value = {'full': self.sample_data}
            
            # Call the method
            results = self.processor.process_directory(test_dir, pattern="*.csv")
            
            # Check that process_file was called for each file
            self.assertEqual(mock_process_file.call_count, 3)
            
            # Check the results
            self.assertEqual(len(results), 3)
    
    def test_merge_datasets(self):
        """Test merging multiple datasets."""
        # Create test datasets
        df1 = pd.DataFrame({
            'id': ['1', '2', '3'],
            'value1': [10, 20, 30]
        })
        df2 = pd.DataFrame({
            'id': ['2', '3', '4'],
            'value2': [200, 300, 400]
        })
        
        # Test simple concatenation
        merged_df = self.processor.merge_datasets([df1, df2])
        self.assertEqual(len(merged_df), len(df1) + len(df2))
        
        # Test merge on column
        merged_df = self.processor.merge_datasets([df1, df2], merge_column='id')
        self.assertEqual(len(merged_df), 4)  # 4 unique IDs
        self.assertIn('value1', merged_df.columns)
        self.assertIn('value2', merged_df.columns)

if __name__ == '__main__':
    unittest.main() 