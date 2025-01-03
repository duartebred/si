import numpy as np

def cosine_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the cosine distance between a single sample and multiple samples.

    Parameters
    ----------
    x : np.ndarray
        A single sample vector.
    y : np.ndarray
        A matrix of multiple sample vectors.
    
    Returns
    -------
    np.ndarray
        An array containing the cosine distances between x and each sample in y.
    """
    # Normalize vectors
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y, axis=1)
    
    # Compute the dot product between x and each row in y
    dot_product = np.dot(y, x)
    
    # Calculate cosine similarity
    similarity = dot_product / (x_norm * y_norm)
    
    # Convert similarity to distance
    return 1 - similarity
