import numpy as np

def cosine_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate the cosine distance between a single point and multiple samples.

    Parameters
    ----------
    x : np.ndarray
        A single data point represented as a vector.
    y : np.ndarray
        A set of data points represented as vectors.

    Returns
    -------
    np.ndarray
        An array with the cosine distances between the input point and each sample.
    """
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y, axis=1)
    
    dot_product = np.dot(y, x)
    
    similarity = dot_product / (x_norm * y_norm)
    
    return 1 - similarity
