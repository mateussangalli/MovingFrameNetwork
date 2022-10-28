import tensorflow as tf
import numpy as np
from data_util import load_dataset, preprocess_and_get_frame, random_rotation
import tensorflow_addons as tfa
import os
import argparse

from get_cnn_model import get_cnn
from weight_decay_scheduler import WeightDecayScheduler


def float2str(a):
    return f'{a:.2f}'.replace('.', 'p')

IMSIZE = 29

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='which dataset to train on')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--lr_decay_freq', type=int, default=20)
parser.add_argument('--id', type=int, default=0)
parser.add_argument('--data_aug', type=str, default='none')

args = parser.parse_args()

(train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = load_dataset(args.dataset)
train_images, train_labels, train_frames = preprocess_and_get_frame(train_images, train_labels, 1.)
val_images, val_labels, val_frames = preprocess_and_get_frame(val_images, val_labels, 1.)
n_classes = np.unique(train_labels).shape[0]



@tf.function
def aug(volumes, labels):
    B = volumes.shape[0]
    volumes = tf.numpy_function(random_rotation, [volumes], [tf.float32])[0]
    volumes = tf.ensure_shape(volumes, [B, IMSIZE, IMSIZE, IMSIZE, 1])
    return volumes, labels


ds_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(train_images.shape[0])
ds_train = ds_train.batch(args.batch_size)
if args.data_aug == 'big':
    ds_train = ds_train.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

config = vars(args)

if args.data_aug == 'big':
    model_name = f'cnn_{args.dataset}_wd{float2str(args.weight_decay)}_augbig_id{args.id}'
elif args.data_aug == 'small':
    model_name = f'cnn_{args.dataset}_wd{float2str(args.weight_decay)}_augsmall_id{args.id}'
else:
    model_name = f'cnn_{args.dataset}_wd{float2str(args.weight_decay)}_id{args.id}'


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

model = get_cnn(n_classes)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tfa.optimizers.AdamW(weight_decay=args.weight_decay, learning_rate=1e-3),
              metrics=['accuracy'])

model.fit(ds_train,
          validation_data=(val_images, val_labels),
          batch_size=args.batch_size,
          epochs=args.epochs,
          verbose=2,
          callbacks=[lr_cb, wd_cb, ckpt_cb])
