# PyAIStatus/PyAIStatus/preprocessing.py

import tensorflow as tf

def create_data_generator(df, image_size=(256, 256), batch_size=32):
    
    # --- THIS IS THE FIX ---
    # The error message told us the model expects 256x256 images.
    print(f"\nCreating data generator with target image size: {image_size}")
    # --- END FIX ---
    
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255  # Normalize pixel values to [0, 1]
    )
    
    generator = datagen.flow_from_dataframe(
        dataframe=df,
        x_col='filepath',
        y_col='label',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # IMPORTANT: Do not shuffle for testing/evaluation
    )
    
    print(f"Found {generator.n} validated image filenames belonging to {len(generator.class_indices)} classes.")
    
    return generator