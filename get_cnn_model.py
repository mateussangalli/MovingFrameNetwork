import tensorflow as tf
from tensorflow.keras import layers


def conv_block(features, n_out):
    features = layers.Conv3D(n_out, (3, 3, 3), use_bias=False, padding='SAME')(features)
    features = layers.BatchNormalization()(features)
    features = layers.Activation(tf.nn.leaky_relu)(features)

    features = layers.Conv3D(n_out, (3, 3, 3), use_bias=False, padding='SAME')(features)
    features = layers.BatchNormalization()(features)
    features = layers.Activation(tf.nn.leaky_relu)(features)
    return features


def get_cnn(n_classes):
    input_volume = layers.Input((None, None, None, 1))

    features = conv_block(input_volume, 16)
    delta = conv_block(features, 16)
    features = layers.Add()([features, delta])

    delta = conv_block(features, 16)
    features = layers.Add()([features, delta])

    features = layers.MaxPooling3D((1, 1, 1), (2, 2, 2))(features)
    features = layers.Conv3D(32, 1, use_bias=False)(features)
    features = layers.BatchNormalization()(features)
    features = layers.Activation(tf.nn.leaky_relu)(features)

    delta = conv_block(features, 32)
    features = layers.Add()([features, delta])

    delta = conv_block(features, 32)
    features = layers.Add()([features, delta])
    features = layers.Conv3D(64, 1, use_bias=False)(features)
    features = layers.BatchNormalization()(features)
    features = layers.Activation(tf.nn.leaky_relu)(features)

    delta = conv_block(features, 64)
    features = layers.Add()([features, delta])
    delta = conv_block(features, 64)
    features = layers.Add()([features, delta])

    features = layers.GlobalMaxPooling3D()(features)
    logits = layers.Dense(n_classes)(features)
    model = tf.keras.models.Model(input_volume, logits)

    return model
