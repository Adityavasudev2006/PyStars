import sys
import platform
import random
import numpy as np
import tensorflow as tf

def get_environment_snapshot() -> dict:
    """Records Python, library versions, and sets a random seed."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    return {
        "python_version": sys.version,
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "random_seed": seed,
    }