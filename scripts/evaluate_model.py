import argparse
from PyAIStatus import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Keras model.")
    parser.add_argument("model_path", type=str, help="Path to the model.h5 file.")
    parser.add_argument("dataset_dir", type=str, help="Path to the dataset directory.")
    parser.add_argument("output_dir", type=str, help="Directory to save the report.")
    args = parser.parse_args()

    evaluate(args.model_path, args.dataset_dir, args.output_dir)