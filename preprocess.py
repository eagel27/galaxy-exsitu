import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from standardize_values import retrieve_mean_std_values
from utilities import log10


def normalize(data):
    normalized_data, norm = tf.linalg.normalize(data, axis=None)
    normalized_data = tf.convert_to_tensor(
        normalized_data)
    print ('----------', norm)
    return normalized_data, norm


def standardize(data):
    scaler = StandardScaler()
    scaler.fit(data)
    normalized_data = scaler.transform(data)
    return normalized_data, scaler


def scale_to_0_1(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    normalized_data = scaler.transform(data)
    return normalized_data


def _per_image_standardization(image):
    """ Linearly scales `image` to have zero mean and unit norm.
    This op computes `(x - mean) / adjusted_stddev`, where `mean` is the average
    of all values in image, and
    `adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements()))`.
    `stddev` is the standard deviation of all values in `image`. It is capped
    away from zero to protect against division by 0 when handling uniform images.
    Args:
    image: 1-D tensor of shape `[height, width]`.
    Returns:
    The standardized image with same shape as `image`.
    Raises:
    ValueError: if the shape of 'image' is incompatible with this function.
    """
    image = ops.convert_to_tensor(image, name='image')
    num_pixels = math_ops.reduce_prod(array_ops.shape(image))

    image = math_ops.cast(image, dtype=dtypes.float32)
    image_mean = math_ops.reduce_mean(image)

    variance = (math_ops.reduce_mean(math_ops.square(image)) -
                math_ops.square(image_mean))
    variance = gen_nn_ops.relu(variance)
    stddev = math_ops.sqrt(variance)
    
    # Apply a minimum normalization that protects us against uniform images.
    min_stddev = math_ops.rsqrt(math_ops.cast(num_pixels, dtypes.float32))
    pixel_value_scale = math_ops.maximum(stddev, min_stddev) + 1e-10
    pixel_value_offset = image_mean

    image = math_ops.subtract(image, pixel_value_offset)
    image = math_ops.div_no_nan(image, pixel_value_scale)
    return image


def _per_image_standardization_div_sum(image):
    image = ops.convert_to_tensor(image, name='image')
    image_sum = math_ops.reduce_sum(image)
    image = math_ops.div_no_nan(image, image_sum)
    return image


def _per_image_standardization_div_minmax(image):
    image = ops.convert_to_tensor(image, name='image')
    image_max = math_ops.reduce_max(image)
    image_min = math_ops.reduce_min(image)
    image = math_ops.div_no_nan(image, image_max - image_min)
    return image


def _per_image_standardization_div_max(image):
    image = ops.convert_to_tensor(image, name='image')
    image_max = math_ops.reduce_max(image) + 1e-10
    image = math_ops.div_no_nan(image, image_max)
    return image


def per_image_standardization(image, channel, logged=False, mode='norm'):
    if logged and channel == 0:
        image = tf.clip_by_value(image, 1e-10, tf.reduce_max(image))
        image = log10(image)
    
    if mode == 'sum':
        norm_im = _per_image_standardization_div_sum(image)
    elif mode == 'divmax':
        norm_im = _per_image_standardization_div_max(image)
    elif mode == 'div_minmax':
        norm_im = _per_image_standardization_div_minmax(image)
    else:
        norm_im = _per_image_standardization(image)
    
    return norm_im


def _per_channel_standardization(image, mean, std):
    """ Works as per_image_standarization,
    but uses mean and std calculated for channel
    """
    image = ops.convert_to_tensor(image, name='image')
    num_pixels = math_ops.reduce_prod(array_ops.shape(image))

    image = math_ops.cast(image, dtype=dtypes.float32)

    # Apply a minimum normalization that protects us against uniform images.
    min_stddev = math_ops.rsqrt(math_ops.cast(num_pixels, dtypes.float32))
    pixel_value_scale = math_ops.maximum(std, min_stddev)
    pixel_value_offset = mean

    image = math_ops.subtract(image, pixel_value_offset)
    image = math_ops.div_no_nan(image, pixel_value_scale)
    image = tf.where(tf.math.is_nan(image), tf.zeros_like(image), image)
    return image


def per_channel_standardization(image, channel, dataset_str, logged=False):
    """ Retrieve mean and std for channel before applying standardization.
    If logged values are required and channel is 0 (Stellar Mass) use
    log_mean and log_std saved in the 3-index of the arrays
    """
    mean_array, std_array = np.float32(retrieve_mean_std_values(dataset_str))
    if logged and channel == 0:
        image = tf.clip_by_value(image, 1e-10, tf.reduce_max(image))
        image = log10(image)
        channel = 3

    mean = mean_array[channel]
    std = std_array[channel]

    return _per_channel_standardization(image, mean, std)


def mask_whole_image(image):
    return tf.zeros(image.shape)


def create_circular_mask(width, height, center=None, radius=None, keep_inner=True):
    """ Return a mask for all pixels outside radius from center """
    if center is None:  # use the middle of the image
        center = (int(width/2), int(height/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], width - center[0], height - center[1])
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    if keep_inner:
        mask = dist_from_center <= radius
    else:
        mask = dist_from_center >= radius
    return mask


def mask_outer_radius(image, radius=128):
    """ Return a copy of the input image with all values
    outside radius masked to zero"""
    height, width = image.shape
    mask_np = create_circular_mask(width, height, radius=radius)
    mask = tf.convert_to_tensor(mask_np, dtype=tf.bool)
    mask = tf.cast(mask, dtype=tf.float32)
    masked_img = mask * image
    return masked_img


def mask_inner_radius(image, radius=128):
    """ Return a copy of the input image with all values
    inside radius masked to zero"""
    height, width = image.shape
    mask_np = create_circular_mask(width, height, radius=radius, keep_inner=False)
    mask = tf.convert_to_tensor(mask_np, dtype=tf.bool)
    mask = tf.cast(mask, dtype=tf.float32)
    masked_img = mask * image
    return masked_img

