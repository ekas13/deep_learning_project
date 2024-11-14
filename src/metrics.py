from scipy.linalg import sqrtm
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub

MNIST_MODULE = "https://tfhub.dev/tensorflow/tfgan/eval/mnist/logits/1"

def fid_score(sampled_data: np.array, real_data: np.array, batch_size: int=64):
    # sampled_data and real_data have to be numpy arrays (samples, 1, 28, 28)
    sampled_data = tf.convert_to_tensor(sampled_data, dtype=tf.float32)
    real_data = tf.convert_to_tensor(real_data, dtype=tf.float32)

    sampled_data = tf.transpose(sampled_data, perm=[0, 2, 3, 1])
    real_data = tf.transpose(real_data, perm=[0, 2, 3, 1])
    
    mnist_classifier_fn = tfhub.load(MNIST_MODULE)
    
    sampled_features_list = []
    real_features_list = []

    with tf.device('/CPU:0'):
        for i in range(0, sampled_data.shape[0], batch_size):
            batch_sampled_data = sampled_data[i:i + batch_size]
            batch_real_data = real_data[i:i + batch_size]
            sampled_features = mnist_classifier_fn(batch_sampled_data)
            real_features = mnist_classifier_fn(batch_real_data)
            sampled_features_list.append(sampled_features)
            real_features_list.append(real_features)

    sampled_features = tf.concat(sampled_features_list, axis=0)
    real_features = tf.concat(real_features_list, axis=0)

    sampled_features = sampled_features.numpy()
    real_features = real_features.numpy()

    mu_sampled = np.mean(sampled_features, axis=0)
    cov_sampled = np.cov(sampled_features, rowvar=False)

    mu_real = np.mean(real_features, axis=0)
    cov_real = np.cov(real_features, rowvar=False)

    mu_diff = mu_sampled - mu_real
    cov_sqrtm = sqrtm(cov_sampled.dot(cov_real))
    return mu_diff.dot(mu_diff) + np.trace(cov_sampled + cov_real - 2 * cov_sqrtm)
