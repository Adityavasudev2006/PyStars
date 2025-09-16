import unittest
import os
import tempfile
import tensorflow as tf
from PyAIStatus import model

class TestModel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        
        self.valid_model_path = os.path.join(self.temp_dir.name, 'dummy_model.h5')
        
        dummy_keras_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(10,)),
            tf.keras.layers.Dense(5, activation='relu', name='dense_layer')
        ])
        
        dummy_keras_model.save(self.valid_model_path)
        
        self.non_existent_model_path = os.path.join(self.temp_dir.name, 'non_existent.h5')

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_model_successfully(self):
        """
        Test that a valid Keras model is loaded correctly.
        """
        # Act: Attempt to load the valid dummy model
        loaded_model = model.load_keras_model(self.valid_model_path)
        
        # Assert: Check that the model object is not None and is a Keras model
        self.assertIsNotNone(loaded_model, "The model should have been loaded, but was None.")
        self.assertIsInstance(loaded_model, tf.keras.Model, "The loaded object is not a Keras Model.")
        
        # Optional: Check for a specific layer to be more certain
        self.assertTrue(any(layer.name == 'dense_layer' for layer in loaded_model.layers),
                        "The loaded model does not contain the expected 'dense_layer'.")

    def test_load_model_file_not_found(self):
        """
        Test that loading a non-existent model returns None.
        """
        # Act: Attempt to load a model from a path that does not exist
        loaded_model = model.load_keras_model(self.non_existent_model_path)
        
        # Assert: Check that the function returns None as expected
        self.assertIsNone(loaded_model, "The function should return None when the model file is not found.")

if __name__ == '__main__':
    unittest.main()