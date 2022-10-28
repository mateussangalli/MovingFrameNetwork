import tensorflow as tf
import numpy as np
from data_util import load_dataset, preprocess_and_get_frame
import os
from scipy.ndimage import rotate
import re
import tensorflow_addons as tfa

from se3_moving_frame import SE3MovingFrame
from se3_movfr_net import get_se3_movfrnet

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='which dataset to test on')
args = parser.parse_args()

N_ANGLES = 24


def float2str(a):
    return f'{a:.2f}'.replace('.', 'p')


def str2float(a):
    return float(a.replace('p', '.'))


def get_sigma_frame(s):
    pos = re.search('sigma', s).span()[1]
    pos += 5
    return str2float(s[pos:pos+4])



def eval_se3movfrnet(model, test_images, test_labels, sigma_frame):
    angles = np.linspace(0, 360., N_ANGLES + 1)[:-1]
    accuracies = np.empty([3, angles.shape[0]])
    for i, angle in enumerate(angles):
        print(f'testing angle {angle}')
        # xy
        new_images = rotate(test_images, angle, (1, 2))
        images, labels, frames = preprocess_and_get_frame(new_images, test_labels, sigma_frame)
        _, acc = model.evaluate([images, frames], labels, verbose=2, batch_size=8)
        accuracies[0, i] = acc

        # xz
        new_images = rotate(test_images, angle, (2, 3))
        images, labels, frames = preprocess_and_get_frame(new_images, test_labels, sigma_frame)
        _, acc = model.evaluate([images, frames], labels, verbose=2, batch_size=8)
        accuracies[1, i] = acc

        # yz
        new_images = rotate(test_images, angle, (3, 1))
        images, labels, frames = preprocess_and_get_frame(new_images, test_labels, sigma_frame)
        _, acc = model.evaluate([images, frames], labels, verbose=2, batch_size=8)
        accuracies[2, i] = acc
    return accuracies


def eval_cnn(model, test_images, test_labels):
    angles = np.linspace(0, 360., N_ANGLES + 1)[:-1]
    accuracies = np.empty([3, angles.shape[0]])
    for i, angle in enumerate(angles):
        print(f'testing angle {angle}')
        # xy
        new_images = rotate(test_images, angle, (1, 2))
        images, labels, frames = preprocess_and_get_frame(new_images, test_labels, 1.)
        _, acc = model.evaluate(images, labels, verbose=2, batch_size=8)
        accuracies[0, i] = acc

        # xz
        new_images = rotate(test_images, angle, (2, 3))
        images, labels, frames = preprocess_and_get_frame(new_images, test_labels, 1.)
        _, acc = model.evaluate(images, labels, verbose=2, batch_size=8)
        accuracies[1, i] = acc

        # yz
        new_images = rotate(test_images, angle, (3, 1))
        images, labels, frames = preprocess_and_get_frame(new_images, test_labels, 1.)
        _, acc = model.evaluate(images, labels, verbose=2, batch_size=8)
        accuracies[2, i] = acc
    return accuracies


models_dir = 'saved_models'
results_dir = 'results'



model_names = os.listdir(models_dir)

(train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = load_dataset(args.dataset)

# cnn models
# pattern = re.compile(f'cnn_{args.dataset}')
# for model_name in filter(lambda s: pattern.match(s), model_names):
#     print(f'testing {model_name}')
#     model = tf.keras.models.load_model(os.path.join(models_dir, model_name))
#     np.save(os.path.join(results_dir, model_name + 'acc'), eval_cnn(model, test_images, test_labels))

# cnn models
pattern = re.compile(f'se3movfrnet.*{args.dataset}')
for model_name in filter(lambda s: pattern.match(s), model_names):
    path = os.path.join(results_dir, model_name + 'acc')
    if os.path.exists(path):
        print(f'skipping {model_name}')
        continue
    print(f'testing {model_name}')
    model = tf.keras.models.load_model(os.path.join(models_dir, model_name))
    np.save(path, eval_se3movfrnet(model, test_images, test_labels, get_sigma_frame(model_name)))
