import tensorflow as tf
from tensorflow.keras import layers

from SE3MovingFrameInvariants import SE3MovingFrameInvariants, Order2Coefficients, Order3Coefficients


class P6EquivariantPooling(layers.Layer):
    def __init__(self, strides=(2, 2, 2), **kwargs):
        super().__init__(**kwargs)

        self.strides = strides

    def call(self, inputs):
        shape = tf.shape(inputs)
        s1 = 2 - (shape[1] % 2)
        s2 = 2 - (shape[2] % 2)
        s3 = 2 - (shape[3] % 2)
        out = tf.nn.max_pool3d(inputs, [s1, s2, s3], self.strides, 'SAME')
        return out

    def get_config(self):
        config = super().get_config().copy()
        config['strides'] = self.strides
        return config


class P6EquivariantFramePooling(layers.Layer):
    def __init__(self, strides=[2, 2, 2], **kwargs):
        super().__init__(**kwargs)

        self.strides = strides

    def call(self, frame):
        shape = tf.shape(frame)
        frame = tf.reshape(frame, [shape[0], shape[1], shape[2], shape[3], 9])
        # s1 = 2 - (shape[1] % 2)
        # s2 = 2 - (shape[2] % 2)
        # s3 = 2 - (shape[3] % 2)
        frame = tf.nn.max_pool3d(frame, [1, 1, 1], self.strides, 'SAME')
        shape = tf.shape(frame)
        frame = tf.reshape(frame, [shape[0], shape[1], shape[2], shape[3], 1, 3, 3])
        return frame

    def get_config(self):
        config = super().get_config().copy()
        config['strides'] = self.strides
        return config


def se3block(features, frame, coefs, order, sigma, width, n_out, dropout):
    features = SE3MovingFrameInvariants(sigma, width, order=order)([features, frame] + coefs)

    features = layers.Conv3D(n_out, 1, use_bias=False)(features)
    features = layers.BatchNormalization()(features)
    features = layers.Activation(tf.nn.leaky_relu)(features)
    features = layers.Dropout(dropout)(features)

    features = layers.Conv3D(n_out, 1, use_bias=False)(features)
    features = layers.BatchNormalization()(features)
    return features


def get_se3_movfrnet(n_classes, order, sigma, dropout):
    input_volume = layers.Input((None, None, None, 1))
    input_frame = layers.Input((None, None, None, 1, 3, 3))
    width = int(4 * sigma + .5)
    coef2 = Order2Coefficients()(input_frame)
    if order > 2:
        coef3 = Order3Coefficients()(input_frame)
        coefs = [coef2, coef3]
    else:
        coefs = [coef2]


    features = se3block(input_volume, input_frame, coefs, order, sigma, width, 16, dropout)
    delta = se3block(features, input_frame, coefs, order, sigma, width, 16, dropout)
    features = layers.Add()([features, delta])

    delta = se3block(features, input_frame, coefs, order, sigma, width, 16, dropout)
    features = layers.Add()([features, delta])

    features = layers.MaxPooling3D((1, 1, 1), (2, 2, 2))(features)
    frame = P6EquivariantFramePooling()(input_frame)

    coef2 = Order2Coefficients()(frame)
    if order > 2:
        coef3 = Order3Coefficients()(frame)
        coefs = [coef2, coef3]
    else:
        coefs = [coef2]

    features = layers.Conv3D(32, 1, use_bias=False)(features)
    features = layers.BatchNormalization()(features)
    features = layers.Activation(tf.nn.leaky_relu)(features)

    delta = se3block(features, frame, coefs, order, sigma, width, 32, dropout)
    features = layers.Add()([features, delta])

    delta = se3block(features, frame, coefs, order, sigma, width, 32, dropout)
    features = layers.Add()([features, delta])
    features = layers.Conv3D(64, 1, use_bias=False)(features)
    features = layers.BatchNormalization()(features)
    features = layers.Activation(tf.nn.leaky_relu)(features)

    delta = se3block(features, frame, coefs, order, sigma, width, 64, dropout)
    features = layers.Add()([features, delta])
    delta = se3block(features, frame, coefs, order, sigma, width, 64, dropout)
    features = layers.Add()([features, delta])

    features = layers.GlobalMaxPooling3D()(features)
    logits = layers.Dense(n_classes)(features)
    model = tf.keras.models.Model([input_volume, input_frame], logits)

    return model
