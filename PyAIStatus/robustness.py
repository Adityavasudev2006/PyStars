# PyAIStatus/PyAIStatus/robustness.py

import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
from . import preprocessing, metrics

# --- Individual Corruption Functions ---
# These functions operate on images in the [0, 1] float format,
# which is what the Keras data generator provides.

def apply_gaussian_noise(image: np.ndarray, sigma=0.1) -> np.ndarray:
    """Adds Gaussian noise to an image."""
    noise = np.random.normal(loc=0, scale=sigma, size=image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0., 1.)

def apply_gaussian_blur(image: np.ndarray, kernel_size=(5, 5)) -> np.ndarray:
    """Applies Gaussian blur to an image."""
    # OpenCV's GaussianBlur works on the [0, 255] range
    img_uint8 = (image * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(img_uint8, kernel_size, 0)
    # Convert back to the [0, 1] float range
    return blurred.astype(np.float32) / 255.0

def apply_jpeg_compression(image: np.ndarray, quality=30) -> np.ndarray:
    """Applies JPEG compression artifact to an image."""
    img_uint8 = (image * 255).astype(np.uint8)
    # Encode to JPEG format in memory and then decode
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_image = cv2.imencode('.jpg', img_uint8, encode_param)
    decoded_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    # OpenCV decodes to BGR, so we convert back to RGB
    decoded_image_rgb = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB)
    return decoded_image_rgb.astype(np.float32) / 255.0

def apply_occlusion(image: np.ndarray, size_fraction=0.2) -> np.ndarray:
    """Applies a random black box occlusion to an image."""
    img_copy = image.copy()
    h, w, _ = img_copy.shape
    
    # Calculate size and position of the occlusion box
    box_h = int(h * size_fraction)
    box_w = int(w * size_fraction)
    # Ensure the box is not placed out of bounds
    start_y = np.random.randint(0, h - box_h + 1)
    start_x = np.random.randint(0, w - box_w + 1)
    
    # Apply the black box
    img_copy[start_y:start_y+box_h, start_x:start_x+box_w, :] = 0
    return img_copy

# --- Main Robustness Testing Function ---

def run_robustness_tests(
    keras_model: tf.keras.Model, 
    test_df: pd.DataFrame, 
    clean_accuracy: float,
    class_names: list
) -> dict:
    """
    Applies various corruptions to the test set and evaluates model performance.

    Args:
        keras_model: The trained Keras model.
        test_df: The DataFrame for the test set.
        clean_accuracy: The model's accuracy on the original, uncorrupted test set.
        class_names: The list of class names for metric calculation.

    Returns:
        A dictionary containing the performance metrics for each corruption type.
    """
    print("\n--- Running Robustness Tests ---")

    corruptions = {
        "Gaussian Noise": apply_gaussian_noise,
        "Gaussian Blur": apply_gaussian_blur,
        "JPEG Compression": apply_jpeg_compression,
        "Occlusion (Random Box)": apply_occlusion
    }
    
    results = {}
    
    # Get the model's expected input size directly from the loaded model
    image_size = keras_model.input_shape[1:3]

    for name, corruption_func in corruptions.items():
        print(f"  Testing with: {name}...")
        
        # Use the existing data generator for consistent preprocessing
        test_generator = preprocessing.create_data_generator(test_df, image_size=image_size)
        
        corrupted_images = []
        # Iterate through the generator to get all original images
        for i in range(len(test_generator)):
            images_batch, _ = test_generator[i]
            for img in images_batch:
                # Apply the corruption function to each image
                corrupted_images.append(corruption_func(img))
        
        # Convert list of corrupted images to a single NumPy array for prediction
        corrupted_images = np.array(corrupted_images)

        # Run inference on the set of corrupted images
        predictions = keras_model.predict(corrupted_images, batch_size=test_generator.batch_size)
        
        # Reuse your existing metrics function for consistent calculations
        corrupted_metrics = metrics.compute_all_metrics(test_generator.classes, predictions, class_names)
        
        # Store the required KPIs for the report
        accuracy = corrupted_metrics.get('overall', {}).get('accuracy', 0)
        f1_score = corrupted_metrics.get('macro_average', {}).get('f1_score', 0)
        
        results[name] = {
            "accuracy": accuracy,
            "macro_f1_score": f1_score,
            "accuracy_delta": accuracy - clean_accuracy # Calculate the performance drop
        }

    print("Robustness tests complete.")
    return results