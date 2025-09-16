# PyAIStatus/PyAIStatus/baseline.py

import tensorflow as tf
import numpy as np
import pandas as pd
import math
from .preprocessing import create_data_generator

RANDOM_SEED = 42

def train_simple_cnn(train_df: pd.DataFrame, 
                     val_df: pd.DataFrame, 
                     image_size: tuple = (150, 150), 
                     batch_size: int = 32,
                     epochs: int = 5) -> tf.keras.Model:
    """Trains a simple, deterministic CNN baseline model."""
    print("\n--- Training Baseline CNN Model ---")
    tf.random.set_seed(RANDOM_SEED)
    train_generator = create_data_generator(train_df, image_size, batch_size)
    val_generator = create_data_generator(val_df, image_size, batch_size)
    num_classes = len(train_generator.class_indices)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(image_size[0], image_size[1], 3)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    print(f"Training for {epochs} epochs...")
    model.fit(train_generator, epochs=epochs, validation_data=val_generator, verbose=1)
    print("Baseline model training complete.")
    return model

def evaluate_baseline(model: tf.keras.Model, 
                      test_df: pd.DataFrame,
                      image_size: tuple = (150, 150),
                      batch_size: int = 32) -> np.ndarray:
    """Evaluates the baseline model using a MANUAL BATCH LOOP for robustness."""
    print("\n--- Evaluating Baseline Model ---")
    test_generator = create_data_generator(test_df, image_size, batch_size)
    
    num_samples = test_generator.n
    steps = math.ceil(num_samples / test_generator.batch_size)
    all_predictions = []

    for i in range(steps):
        images, _ = next(test_generator)
        batch_predictions = model.predict_on_batch(images)
        all_predictions.append(batch_predictions)

    y_pred_proba = np.vstack(all_predictions)
    y_pred_proba = y_pred_proba[:num_samples]

    print(f"Baseline evaluation complete. Got {len(y_pred_proba)} predictions.")
    return y_pred_proba