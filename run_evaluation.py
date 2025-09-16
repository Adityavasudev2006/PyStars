# code.py

import os
from PyAIStatus import evaluate

#Defining Location
model_location = os.path.join("models", "Dogs-vs-Cats_model.h5")
data_location = os.path.join("data", "cats_and_dogs_dataset", "train")
output_location = os.path.join("evaluation_results")

#Calling the evaluate function
if not os.path.exists(model_location):
    print(f"Error: Model file not found at '{model_location}'")
elif not os.path.exists(data_location):
    print(f"Error: Dataset directory not found at '{data_location}'")
else:
    evaluate(
        model_path=model_location,
        dataset_dir=data_location,
        output_dir=output_location
    )