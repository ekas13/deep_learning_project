from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.image import resize
from scipy.linalg import sqrtm
import numpy as np
import tensorflow as tf

def fid_score_cifar10_resized(sampled_data: np.array, real_data: np.array, batch_size: int=64):
    """
    Calculate FID score for CIFAR-10 data after resizing to meet InceptionV3 input requirements.
    
    Parameters:
    - sampled_data: Generated samples as a numpy array of shape (num_samples, 3, 32, 32).
    - real_data: Real CIFAR-10 samples as a numpy array of shape (num_samples, 3, 32, 32).
    - batch_size: Batch size for feature extraction.

    Returns:
    - FID score (float).
    """
    # Ensure data is in the format TensorFlow expects (num_samples, 32, 32, 3)
    if sampled_data.shape[1:] != (3, 32, 32) or real_data.shape[1:] != (3, 32, 32):
        raise ValueError("Input data must have shape (num_samples, 3, 32, 32).")

    # Transpose to channel-last format
    sampled_data = np.transpose(sampled_data, (0, 2, 3, 1))  # (num_samples, 32, 32, 3)
    real_data = np.transpose(real_data, (0, 2, 3, 1))        # (num_samples, 32, 32, 3)

    # Load InceptionV3 model
    inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(75, 75, 3))

    # Preprocess and resize data
    def resize_and_process(data):
        resized = np.array([resize(img, (75, 75)).numpy() for img in data])  # Resize to (75, 75, 3)
        return preprocess_input(resized)  # Normalize images for InceptionV3

    sampled_data_resized = resize_and_process(sampled_data)
    real_data_resized = resize_and_process(real_data)

    # Extract features
    def get_features(data):
        num_samples = data.shape[0]
        features_list = []
        for i in range(0, num_samples, batch_size):
            batch_data = data[i:i + batch_size]
            features = inception_model(batch_data)
            features_list.append(features)
        return tf.concat(features_list, axis=0).numpy()

    sampled_features = get_features(sampled_data_resized)
    real_features = get_features(real_data_resized)

    # Compute statistics
    mu_sampled = np.mean(sampled_features, axis=0)
    cov_sampled = np.cov(sampled_features, rowvar=False)

    mu_real = np.mean(real_features, axis=0)
    cov_real = np.cov(real_features, rowvar=False)

    # Compute FID score
    mu_diff = mu_sampled - mu_real
    cov_sqrtm = sqrtm(cov_sampled.dot(cov_real))
    if np.iscomplexobj(cov_sqrtm):
        cov_sqrtm = cov_sqrtm.real
    fid = mu_diff.dot(mu_diff) + np.trace(cov_sampled + cov_real - 2 * cov_sqrtm)
    return fid
