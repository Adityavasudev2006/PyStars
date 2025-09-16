# code.py

import os
from PyAIStatus import evaluate

# --- Define the paths to your resources ---
# Using os.path.join is a good practice to make paths work on any OS.
model_location = os.path.join("models", "Dogs-vs-Cats_model.h5")
data_location = os.path.join("data", "cats_and_dogs_dataset", "train")
output_location = os.path.join("evaluation_results") # Use a new folder to see the result

# --- Check if the paths exist before running ---
if not os.path.exists(model_location):
    print(f"Error: Model file not found at '{model_location}'")
elif not os.path.exists(data_location):
    print(f"Error: Dataset directory not found at '{data_location}'")
else:
    # --- Call your library's evaluate function directly ---
    # Note the argument names match the function definition.
    evaluate(
        model_path=model_location,
        dataset_dir=data_location,
        output_dir=output_location
    )