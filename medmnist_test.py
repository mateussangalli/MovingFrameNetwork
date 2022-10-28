import tensorflow as tf
import numpy as np
from data_util import load_dataset, preprocess_and_get_frame
import tensorflow_addons as tfa
import os
from scipy.ndimage import rotate

from se3_moving_frame import SE3MovingFrame
from se3_movfr_net import get_se3_movfrnet

import argparse


def float2str(a):
    return f'{a:.2f}'.replace('.', 'p')


parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='which dataset to train on')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--sigma', type=float, default=1.)
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--lr_decay_freq', type=int, default=20)
parser.add_argument('--id', type=int, default=0)
parser.add_argument('--order', type=int, default=2)
parser.add_argument('--sigma_frame', type=float, default=1.)

args = parser.parse_args()

(train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = load_dataset(args.dataset)

config = vars(args)
model_name = f'se3movfrnet_order{args.order}_{args.dataset}_sigma{float2str(args.sigma)}_{float2str(args.sigma_frame)}_dropout{float2str(args.dropout)}_wd{float2str(args.weight_decay)}_id{args.id}'

model = tf.keras.models.load_model(os.path.join('saved_models', model_name))

angles = np.linspace(0, 360., 25)[:-1]

accuracies = np.empty([3, angles.shape[0]])
for i, angle in enumerate(angles):
    print(f'testing angle {angle}')
    # xy
    new_images = rotate(test_images, angle, (1, 2))
    images, labels, frames = preprocess_and_get_frame(new_images, test_labels, args.sigma_frame)
    _, acc = model.evaluate([images, frames], labels, verbose=2)
    accuracies[0, i] = acc

    # xz
    new_images = rotate(test_images, angle, (2, 3))
    images, labels, frames = preprocess_and_get_frame(new_images, test_labels, args.sigma_frame)
    _, acc = model.evaluate([images, frames], labels, verbose=2)
    accuracies[1, i] = acc

    # yz
    new_images = rotate(test_images, angle, (3, 1))
    images, labels, frames = preprocess_and_get_frame(new_images, test_labels, args.sigma_frame)
    _, acc = model.evaluate([images, frames], labels, verbose=2)
    accuracies[2, i] = acc
print(accuracies)
np.save(os.path.join('results', model_name + 'acc'), accuracies)
