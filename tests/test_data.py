# In tests/test_data.py

import unittest
import os
import tempfile
import pandas as pd
from PyAIStatus import data

class TestData(unittest.TestCase):
    def setUp(self):
        """Set up a temporary directory with a mock dataset."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dataset_path = self.temp_dir.name

        # Create a more robust mock dataset directory structure
        # 10 samples total is enough to avoid stratification errors with a 20% split
        os.makedirs(os.path.join(self.test_dataset_path, 'class_happy'), exist_ok=True)
        os.makedirs(os.path.join(self.test_dataset_path, 'class_sad'), exist_ok=True)

        for i in range(5):
            with open(os.path.join(self.test_dataset_path, 'class_happy', f'happy_{i}.jpg'), 'w') as f:
                f.write('dummy image')
            with open(os.path.join(self.test_dataset_path, 'class_sad', f'sad_{i}.jpg'), 'w') as f:
                f.write('dummy image')

    def tearDown(self):
        """Clean up the temporary directory."""
        self.temp_dir.cleanup()

    def test_get_dataset_summary(self):
        """Test the dataset summary function for correct class names and counts."""
        class_names, image_counts = data.get_dataset_summary(self.test_dataset_path)
        self.assertEqual(len(class_names), 2)
        self.assertEqual(len(image_counts), 2)
        # Sort to ensure consistent order for comparison
        sorted_summary = sorted(zip(class_names, image_counts))
        self.assertEqual(sorted_summary[0][0], 'class_happy')
        self.assertEqual(sorted_summary[0][1], 5)
        self.assertEqual(sorted_summary[1][0], 'class_sad')
        self.assertEqual(sorted_summary[1][1], 5)

    def test_split_data_returns_dataframes(self):
        """Test that the split function returns two pandas DataFrames."""
        train_df, test_df = data.split_data(self.test_dataset_path, test_size=0.2)
        self.assertIsInstance(train_df, pd.DataFrame)
        self.assertIsInstance(test_df, pd.DataFrame)
        self.assertIn('filepath', train_df.columns)
        self.assertIn('label', train_df.columns)

    def test_split_data_correct_size(self):
        """Test that the split is the correct size."""
        train_df, test_df = data.split_data(self.test_dataset_path, test_size=0.2)
        # 10 total samples, 20% test size -> 2 for test, 8 for train
        self.assertEqual(len(train_df), 8)
        self.assertEqual(len(test_df), 2)

if __name__ == '__main__':
    unittest.main()