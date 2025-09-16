# PyAIStatus/PyAIStatus/explainability.py

import numpy as np
import cv2
import tensorflow as tf
import base64
from io import BytesIO

try:
    from tf_keras_vis.saliency import Saliency
    from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
    from tf_keras_vis.utils.scores import CategoricalScore
    TF_KERAS_VIS_AVAILABLE = True
except ImportError:
    print("Warning: tf-keras-vis library not found. Explainability features will be skipped.")
    TF_KERAS_VIS_AVAILABLE = False

# --- Helper Functions ---

def _overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    img_uint8 = (image * 255).astype(np.uint8)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlaid_image = cv2.addWeighted(img_uint8, 1 - alpha, colored_heatmap, alpha, 0)
    return overlaid_image

def _save_image_to_base64(image_array: np.ndarray) -> str:
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', image_bgr)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"

# --- Core Explainability Functions ---

def _generate_attribution_maps(model: tf.keras.Model, data_generator, class_names: list) -> dict:
    print("  Generating attribution maps (Grad-CAM)...")
    attribution_maps = {}
    
    replace2linear = ReplaceToLinear()
    saliency = Saliency(model, model_modifier=replace2linear, clone=True)
    is_binary_output = model.output_shape[-1] == 1

    found_samples = {name: None for name in class_names}
    for i in range(len(data_generator)):
        images_batch, labels_batch = data_generator[i]
        int_labels = np.argmax(labels_batch, axis=1)
        for img, label_idx in zip(images_batch, int_labels):
            class_name = class_names[label_idx]
            if found_samples[class_name] is None:
                found_samples[class_name] = img
        if all(v is not None for v in found_samples.values()):
            break

    for class_name, image_to_explain in found_samples.items():
        if image_to_explain is None:
            print(f"    Warning: Could not find a sample for class '{class_name}'.")
            continue
        
        if is_binary_output:
            score = CategoricalScore([0])
        else:
            class_index = class_names.index(class_name)
            score = CategoricalScore([class_index])
        
        saliency_map = saliency(score, np.expand_dims(image_to_explain, axis=0))
        
        overlay = _overlay_heatmap(image_to_explain, saliency_map[0])
        attribution_maps[class_name] = _save_image_to_base64(overlay)

    return attribution_maps

def _calculate_stability_score(model: tf.keras.Model, data_generator, class_names: list) -> dict:
    print("  Calculating stability scores...")
    stability_scores = {}
    replace2linear = ReplaceToLinear()
    
    # --- THIS IS THE CORRECTED LINE ---
    # The 'method' argument is removed. The function will default to Grad-CAM.
    saliency = Saliency(model, model_modifier=replace2linear, clone=True)
    is_binary_output = model.output_shape[-1] == 1
    
    samples_per_class = {name: [] for name in class_names}
    num_samples_needed = 3
    for i in range(len(data_generator)):
        images_batch, labels_batch = data_generator[i]
        int_labels = np.argmax(labels_batch, axis=1)
        for img, label_idx in zip(images_batch, int_labels):
            class_name = class_names[label_idx]
            if len(samples_per_class[class_name]) < num_samples_needed:
                samples_per_class[class_name].append(img)
        if all(len(v) == num_samples_needed for v in samples_per_class.values()):
            break

    for class_name, images in samples_per_class.items():
        if not images: continue
        if is_binary_output:
            score = CategoricalScore([0])
        else:
            class_index = class_names.index(class_name)
            score = CategoricalScore([class_index])

        class_scores = []
        for img in images:
            original_map = saliency(score, np.expand_dims(img, axis=0))
            map_differences = []
            for _ in range(5):
                noise = np.random.normal(0, 0.1, img.shape)
                noisy_img = np.clip(img + noise, 0, 1)
                noisy_map = saliency(score, np.expand_dims(noisy_img, axis=0))
                diff = np.mean(np.abs(original_map - noisy_map))
                map_differences.append(diff)
            class_scores.append(np.mean(map_differences))
            
        stability_scores[class_name] = np.mean(class_scores)
        
    return stability_scores

def run_explainability_tasks(model: tf.keras.Model, test_generator, class_names: list) -> dict:
    """Orchestrates all explainability tasks and returns a results dictionary."""
    print("\n--- Running Explainability Tasks ---")
    
    if not TF_KERAS_VIS_AVAILABLE:
        return {
            "attribution_maps": {"error": "tf-keras-vis is not installed. Skipping."},
            "stability_scores": {"error": "tf-keras-vis is not installed. Skipping."},
            "overlap_coefficient_note": "Not applicable: Ground-truth segmentation masks are not available for this dataset."
        }
    
    results = {
        "attribution_maps": _generate_attribution_maps(model, test_generator, class_names),
        "stability_scores": _calculate_stability_score(model, test_generator, class_names),
        "overlap_coefficient_note": "Not applicable: Ground-truth segmentation masks are not available for this dataset."
    }
    
    print("Explainability tasks complete.")
    return results