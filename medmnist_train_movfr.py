import tensorflow as tf
import numpy as np
from data_util import load_dataset, preprocess_and_get_frame
import tensorflow_addons as tfa
import os

from se3_moving_frame import SE3MovingFrame
from se3_movfr_net import get_se3_movfrnet
from weight_decay_scheduler import WeightDecayScheduler

def float2str(a):
    return f'{a:.2f}'.replace('.', 'p')

import argparse

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
train_images, train_labels, train_frames = preprocess_and_get_frame(train_images, train_labels, args.sigma_frame)
val_images, val_labels, val_frames = preprocess_and_get_frame(val_images, val_labels, args.sigma_frame)
n_classes = np.unique(train_labels).shape[0]

config = vars(args)

model_name = f'se3movfrnet_order{args.order}_{args.dataset}_sigma{float2str(args.sigma)}_{float2str(args.sigma_frame)}_dropout{float2str(args.dropout)}_wd{float2str(args.weight_decay)}_id{args.id}'

def lr_schedule(epoch, lr):
    if (epoch + 1) % args.lr_decay_freq == 0:
        return lr * .1
    else:
        return lr

lr_cb = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
wd_cb = WeightDecayScheduler(lr_schedule, verbose=1)

ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
    os.path.join('saved_models', model_name),
    monitor='val_accuracy',
    save_best_only=True
)

model = get_se3_movfrnet(n_classes, args.order, args.sigma, args.dropout)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tfa.optimizers.AdamW(weight_decay=args.weight_decay, learning_rate=1e-3),
              metrics=['accuracy'])


model.fit([train_images, train_frames], train_labels,
          validation_data=([val_images, val_frames], val_labels),
          batch_size=args.batch_size,
          epochs=args.epochs,
          verbose=2,
          callbacks=[lr_cb, wd_cb, ckpt_cb])
