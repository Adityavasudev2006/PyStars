# PyAIStatus/PyAIStatus/model.py

import tensorflow as tf
import numpy as np
import math

def load_keras_model(model_path: str) -> 'tf.keras.Model':
    """Loads a Keras model from a .h5 file."""
    print(f"\nAttempting to load model from: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Model loaded successfully.")
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        summary_str = "\n".join(summary_lines)
        print(summary_str)
        
        # Return both the model and its summary
        return model, summary_str
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def run_inference(model: 'tf.keras.Model', test_generator) -> np.ndarray:
    """
    Runs model inference using a MANUAL BATCH LOOP for maximum robustness.
    This bypasses unpredictable behavior in model.predict(generator).
    """
    print("\nStarting model inference with MANUAL BATCH LOOP...")
    
    try:
        num_samples = test_generator.n
        batch_size = test_generator.batch_size
        steps = math.ceil(num_samples / batch_size)
        
        all_predictions = []
        
        print(f"Manually iterating for {steps} steps to process {num_samples} samples...")
        
        for i in range(steps):
            # Get the next batch of images and labels from the generator
            images, _ = next(test_generator)
            # Run prediction on just this batch
            batch_predictions = model.predict_on_batch(images)
            all_predictions.append(batch_predictions)
            
        # Combine the predictions from all batches into one array
        y_pred_proba = np.vstack(all_predictions)
        
        # Slice the results to ensure it's exactly the number of samples
        y_pred_proba = y_pred_proba[:num_samples]

        print(f"Inference complete. Processed {len(y_pred_proba)} samples.")
        
        return y_pred_proba
        
    except Exception as e:
        print(f"An error occurred during model prediction: {e}")
        return None