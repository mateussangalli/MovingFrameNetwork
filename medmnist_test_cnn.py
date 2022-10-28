import tensorflow as tf
import numpy as np
from data_util import load_dataset, preprocess_and_get_frame
import tensorflow_addons as tfa
import os
from scipy.ndimage import rotate

from get_cnn_model import get_cnn



def float2str(a):
    return f'{a:.2f}'.replace('.', 'p')


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='which dataset to train on')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--sigma', type=float, default=1.)
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--lr_decay_freq', type=int, default=15)
parser.add_argument('--id', type=int, default=0)

args = parser.parse_args()

(train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = load_dataset(args.dataset)

config = vars(args)
model_name = f'cnn_{args.dataset}_wd{float2str(args.weight_decay)}_id{args.id}'

model = tf.keras.models.load_model(os.path.join('saved_models', model_name))

angles = np.linspace(0, 360., 25)[:-1]

accuracies = np.empty([3, angles.shape[0]])
for i, angle in enumerate(angles):
    print(f'testing angle {angle}')
    #xy
    new_images = rotate(test_images, angle, (1, 2))
    images, labels, frames = preprocess_and_get_frame(new_images, test_labels, 1)
    _, acc = model.evaluate(images, labels, verbose=2)
    accuracies[0, i] = acc

    #xz
    new_images = rotate(test_images, angle, (2, 3))
    images, labels, frames = preprocess_and_get_frame(new_images, test_labels, 1)
    _, acc = model.evaluate(images, labels, verbose=2)
    accuracies[1, i] = acc

    #yz
    new_images = rotate(test_images, angle, (3, 1))
    images, labels, frames = preprocess_and_get_frame(new_images, test_labels, 1)
    _, acc = model.evaluate(images, labels, verbose=2)
    accuracies[2, i] = acc
print(accuracies)
np.save(os.path.join('results', model_name + 'acc'), accuracies)
