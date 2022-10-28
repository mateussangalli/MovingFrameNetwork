import numpy as np
import os
from skimage.transform import resize
from se3_moving_frame import SE3MovingFrame

from scipy.ndimage import rotate
from scipy.spatial.transform import Rotation


def random_rotation(vol):
    angles = Rotation.random(vol.shape[0]).as_euler('zxy', degrees=True)[0, :]
    out = rotate(vol, angles[0], (1, 2), reshape=False)
    out = rotate(out, angles[1], (2, 3), reshape=False)
    out = rotate(out, angles[2], (3, 1), reshape=False)
    return out.astype(np.float32)


def load_dataset(dataset_name, path='medmnist/'):
    dataset = np.load(os.path.join(path, dataset_name + '.npz'))

    train_images = dataset['train_images']
    train_labels = dataset['train_labels']
    val_images = dataset['val_images']
    val_labels = dataset['val_labels']
    test_images = dataset['test_images']
    test_labels = dataset['test_labels']
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)


def preprocess_and_get_frame(images, labels, sigma, threshold=1e-5, size=29):
    images = resize(images, [images.shape[0], size, size, size])
    images = images[..., np.newaxis]
    labels = labels[:, 0]

    moving_frame_gen = SE3MovingFrame(sigma, threshold=threshold)
    frames = np.empty(images.shape + (3, 3))
    for i in range(images.shape[0]):
        frames[i, ...] = moving_frame_gen(images[i, np.newaxis, ..., 0])
    return images, labels, frames
