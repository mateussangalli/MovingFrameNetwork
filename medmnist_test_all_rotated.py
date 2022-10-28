import tensorflow as tf
import numpy as np
from data_util import load_dataset, load_dataset_rotated, preprocess_and_get_frame
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



models_dir = 'saved_models'
results_dir = 'results'



model_names = os.listdir(models_dir)

test_images, test_labels = load_dataset_rotated(args.dataset)
# _, (test_images, test_labels), _ = load_dataset(args.dataset)

results = {}


# cnn models
pattern = re.compile(f'cnn_{args.dataset}')
for model_name in filter(lambda s: pattern.match(s), model_names):
    print(f'testing {model_name}')
    model = tf.keras.models.load_model(os.path.join(models_dir, model_name))
    images, labels, frames = preprocess_and_get_frame(test_images, test_labels, 1.)
    _, acc = model.evaluate(images, labels)
    print(model_name)
    print(acc)
    results[model_name] = acc

# se3movf models
pattern = re.compile(f'se3movfrnet.*{args.dataset}')
for model_name in filter(lambda s: pattern.match(s), model_names):
    print(f'testing {model_name}')
    sigma_frame = get_sigma_frame(model_name)
    model = tf.keras.models.load_model(os.path.join(models_dir, model_name))
    images, labels, frames = preprocess_and_get_frame(test_images, test_labels, sigma_frame)
    _, acc = model.evaluate([images, frames], labels)
    print(model_name)
    print(acc)
    results[model_name] = acc

np.savez_compressed(f'results_{args.dataset}', **results)
