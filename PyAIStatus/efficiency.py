# PyAIStatus/PyAIStatus/efficiency.py

import os
import time
import tracemalloc
import numpy as np
import tensorflow as tf

def get_efficiency_metrics(model_path: str, batch_size_for_timing: int = 32) -> dict:
    """
    Calculates and reports all required efficiency metrics for a given model (Task 15).

    Args:
        model_path: The file path to the .h5 model file.
        batch_size_for_timing: The batch size to use for measuring inference speed.

    Returns:
        A dictionary containing the calculated efficiency metrics.
    """
    print("\nCalculating efficiency metrics...")

    # --- 1. Model File Size ---
    try:
        file_size_bytes = os.path.getsize(model_path)
    except OSError as e:
        print(f"  Could not get file size: {e}")
        file_size_bytes = -1

    # --- 2. Parameter Count, Inference Time & 4. Peak Memory ---
    # These require loading the model.
    param_count = -1
    avg_inference_time_ms = -1
    peak_memory_mb = -1

    try:
        # --- 4. Peak Memory Usage (Start tracing) ---
        tracemalloc.start()

        # Load the model to get its properties
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Get current and peak memory since tracing started
        current, peak = tracemalloc.get_traced_memory()
        peak_memory_mb = peak / 10**6  # Convert bytes to megabytes
        tracemalloc.stop()

        # --- 2. Parameter Count ---
        param_count = model.count_params()

        # --- 3. Average Inference Time ---
        # Get the input shape from the model's first layer
        input_shape = model.input_shape
        # Create a batch of dummy data with the correct shape (e.g., (32, 224, 224, 3))
        dummy_input_shape = (batch_size_for_timing,) + input_shape[1:]
        dummy_data = np.random.rand(*dummy_input_shape).astype(np.float32)

        # "Warm-up" run: The first prediction can be slower due to initialization.
        _ = model.predict(dummy_data, verbose=0)

        # Time the inference over several runs for a more stable average
        num_runs = 10
        start_time = time.perf_counter()
        for _ in range(num_runs):
            _ = model.predict(dummy_data, verbose=0)
        end_time = time.perf_counter()

        total_time_s = end_time - start_time
        # Calculate time per image in milliseconds
        avg_inference_time_ms = (total_time_s / (num_runs * batch_size_for_timing)) * 1000

    except Exception as e:
        print(f"  Could not load model to calculate parameters, timing, or memory: {e}")
    
    efficiency_results = {
        "model_file_size_bytes": file_size_bytes,
        "total_parameter_count": param_count,
        "average_inference_time_ms_per_image": avg_inference_time_ms,
        "peak_memory_usage_mb": peak_memory_mb
    }

    print("  Efficiency metrics calculated:")
    print(f"    - File Size: {file_size_bytes / 1e6:.2f} MB")
    print(f"    - Parameter Count: {param_count:,}")
    print(f"    - Inference Time: {avg_inference_time_ms:.4f} ms/image")
    print(f"    - Peak Memory: {peak_memory_mb:.2f} MB")

    return efficiency_results