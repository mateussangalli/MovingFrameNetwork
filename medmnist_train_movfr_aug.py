import tensorflow as tf
import numpy as np
from data_util import load_dataset, preprocess_and_get_frame, random_rotation
import tensorflow_addons as tfa
import os

from se3_moving_frame import SE3MovingFrame
from se3_movfr_net import get_se3_movfrnet
from weight_decay_scheduler import WeightDecayScheduler

import argparse

def float2str(a):
    return f'{a:.2f}'.replace('.', 'p')

IMSIZE = 29

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
parser.add_argument('--aug', type=str, default='big')

args = parser.parse_args()

moving_frame_gen = SE3MovingFrame(args.sigma_frame, threshold=1e-5)


def get_frame(volumes):
    frames = np.empty(volumes.shape + (3, 3), dtype=np.float32)
    for i in range(volumes.shape[0]):
        frames[i, ...] = moving_frame_gen(volumes[i, np.newaxis, ..., 0])
    return frames


@tf.function
def aug(volumes, labels):
    B = volumes.shape[0]
    volumes = tf.numpy_function(random_rotation, [volumes], [tf.float32])[0]
    volumes = tf.ensure_shape(volumes, [B, IMSIZE, IMSIZE, IMSIZE, 1])
    frames = tf.numpy_function(get_frame, [volumes], [tf.float32])[0]
    frames = tf.ensure_shape(frames, [B, IMSIZE, IMSIZE, IMSIZE, 1, 3, 3])
    return (volumes, frames), labels

(train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = load_dataset(args.dataset)
train_images, train_labels, train_frames = preprocess_and_get_frame(train_images, train_labels, args.sigma_frame)
val_images, val_labels, val_frames = preprocess_and_get_frame(val_images, val_labels, args.sigma_frame)
n_classes = np.unique(train_labels).shape[0]

ds_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(train_images.shape[0])
ds_train = ds_train.batch(args.batch_size)
ds_train = ds_train.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

config = vars(args)

model_name = f'se3movfrnet_order{args.order}_{args.dataset}_sigma{float2str(args.sigma)}_{float2str(args.sigma_frame)}_dropout{float2str(args.dropout)}_wd{float2str(args.weight_decay)}_aug{args.aug}_id{args.id}'


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
