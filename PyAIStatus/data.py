# PyAIStatus/PyAIStatus/data.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split

def get_dataset_summary(dataset_dir: str) -> tuple:
    """Creates a table of class names and image counts."""
    class_names = []
    image_counts = []

    if not os.path.isdir(dataset_dir):
        print(f"Error: The provided dataset directory does not exist: {dataset_dir}")
        return None, None

    for item in sorted(os.listdir(dataset_dir)):
        item_path = os.path.join(dataset_dir, item)
        if os.path.isdir(item_path):
            num_images = len(os.listdir(item_path))
            if num_images > 0:
                class_names.append(item)
                image_counts.append(num_images)

    if not class_names:
        print(f"Error: No subdirectories with images found in {dataset_dir}")
        return None, None

    print("Dataset Summary:")
    print(f"{'Class Name':<20} | {'Image Count'}")
    print("-" * 35)
    for name, count in zip(class_names, image_counts):
        print(f"{name:<20} | {count}")

    return class_names, image_counts


def split_data(dataset_dir: str, test_size: float = 0.2, seed: int = 42) -> tuple:
    """
    Scans the dataset directory, creates a DataFrame of filepaths and labels,
    and splits it into stratified train and test sets.
    """
    filepaths = []
    labels = []
    class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])

    if not class_names:
        print(f"Error: No class subdirectories found in {dataset_dir}")
        return None, None

    for label in class_names:
        class_dir = os.path.join(dataset_dir, label)
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                filepaths.append(os.path.join(class_dir, filename))
                labels.append(label)

    if not filepaths:
        print(f"Error: No image files found in the subdirectories of {dataset_dir}")
        return None, None

    df = pd.DataFrame({'filepath': filepaths, 'label': labels})

    # Stratify ensures the same proportion of classes in train and test sets
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df['label']
    )

    print(f"\nData split:")
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")
    print(f"Split ratio (test size): {test_size}, Seed: {seed}")

    return train_df, test_df