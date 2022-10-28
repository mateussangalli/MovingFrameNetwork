import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from scipy.special import eval_hermitenorm

@tf.function
def depthconvx(inputs, kernel):
    shape = tf.shape(inputs)
    out = tf.reshape(inputs, [shape[0], shape[1], -1, shape[4]])
    out = tf.nn.depthwise_conv2d(out, kernel, (1, 1, 1, 1), 'SAME')
    out = tf.reshape(out, shape)
    return out


@tf.function
def depthconvy(inputs, kernel):
    out = tf.transpose(inputs, [0, 2, 1, 3, 4])
    shape = tf.shape(out)
    out = tf.reshape(out, [shape[0], shape[1], -1, shape[4]])
    out = tf.nn.depthwise_conv2d(out, kernel, (1, 1, 1, 1), 'SAME')
    out = tf.reshape(out, shape)
    out = tf.transpose(out, [0, 2, 1, 3, 4])
    return out


@tf.function
def depthconvz(inputs, kernel):
    out = tf.transpose(inputs, [0, 3, 1, 2, 4])
    shape = tf.shape(out)
    out = tf.reshape(out, [shape[0], shape[1], -1, shape[4]])
    out = tf.nn.depthwise_conv2d(out, kernel, (1, 1, 1, 1), 'SAME')
    out = tf.reshape(out, shape)
    out = tf.transpose(out, [0, 2, 3, 1, 4])
    return out


class SE3MovingFrame(layers.Layer):
    def __init__(self, sigma, threshold=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.sigma = sigma
        width = int(np.round(4 * sigma))
        self.width = width


        x = np.arange(-width, width + 1, dtype=np.float32)
        g0 = np.exp(-(x ** 2) / (2 * sigma ** 2))
        g0 /= np.sqrt(2 * np.pi * sigma)

        g = [g0]
        for n in range(1, 3):
            tmp = (1 - 2 * (n % 2)) * eval_hermitenorm(n, x / sigma) * g0
            tmp /= sigma ** n
            g.append(tmp)

        self.g = [p.reshape([-1, 1, 1, 1]).astype(np.float32) for p in g]

    def build(self, input_shape):
        if input_shape[-1] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        n_channels = input_shape[-1]
        self.n_channels = n_channels
        self.g = [np.concatenate(n_channels * [p], 2) for p in self.g]
        self.g = [tf.constant(p, dtype=tf.float32) for p in self.g]

        super().build(input_shape)

    def _get_derivatives_order1(self, inputs):
        ux = depthconvx(inputs, self.g[1])
        ux = depthconvy(ux, self.g[0])
        ux = depthconvz(ux, self.g[0])

        uy = depthconvx(inputs, self.g[0])
        uy = depthconvy(uy, self.g[1])
        uy = depthconvz(uy, self.g[0])

        uz = depthconvx(inputs, self.g[0])
        uz = depthconvy(uz, self.g[0])
        uz = depthconvz(uz, self.g[1])

        return ux, uy, uz

    def _get_derivatives_order2(self, inputs):
        uxx = depthconvx(inputs, self.g[2])
        uxx = depthconvy(uxx, self.g[0])
        uxx = depthconvz(uxx, self.g[0])

        uyy = depthconvx(inputs, self.g[0])
        uyy = depthconvy(uyy, self.g[2])
        uyy = depthconvz(uyy, self.g[0])

        uzz = depthconvx(inputs, self.g[0])
        uzz = depthconvy(uzz, self.g[0])
        uzz = depthconvz(uzz, self.g[2])

        uxy = depthconvx(inputs, self.g[1])
        uxy = depthconvy(uxy, self.g[1])
        uxy = depthconvz(uxy, self.g[0])

        uxz = depthconvx(inputs, self.g[1])
        uxz = depthconvy(uxz, self.g[0])
        uxz = depthconvz(uxz, self.g[1])

        uyz = depthconvx(inputs, self.g[0])
        uyz = depthconvy(uyz, self.g[1])
        uyz = depthconvz(uyz, self.g[1])
        return uxx, uyy, uzz, uxy, uxz, uyz

    def call(self, inputs):
        uxx, uyy, uzz, uxy, uxz, uyz = self._get_derivatives_order2(inputs)
        ux, uy, uz = self._get_derivatives_order1(inputs)
        grad = tf.stack([ux, uy, uz], -1)

        hess1 = tf.stack([uxx, uxy, uxz], -1)
        hess2 = tf.stack([uxy, uyy, uyz], -1)
        hess3 = tf.stack([uxz, uyz, uzz], -1)
        hess = tf.stack([hess1, hess2, hess3], -1)
        l, v = tf.linalg.eigh(hess)
        l = l[:, :, :, :, :, ::-1]
        v = v[:, :, :, :, :, :, ::-1]

        v0 = v[:, :, :, :, :, :, 0]
        v1 = v[:, :, :, :, :, :, 1]
        v2 = v[:, :, :, :, :, :, 2]
        # dot0 = 0
        # dot1 = 0
        dot0 = tf.reduce_sum(v0 * grad, -1, keepdims=True)
        dot1 = tf.reduce_sum(v1 * grad, -1, keepdims=True)


        v0 = tf.where(dot0 < 0, -v0, v0)
        v1 = tf.where(dot1 < 0, -v1, v1)
        v = tf.stack([v0, v1, v2], -1)

        v2 = tf.where(tf.expand_dims(tf.linalg.det(v), 5) < 0, -v2, v2)
        v = tf.stack([v0, v1, v2], -1)

        mask = tf.abs(l[:, :, :, :, :, 0] - l[:, :, :, :, :, 1])
        mask = tf.minimum(mask, tf.abs(l[:, :, :, :, :, 1] - l[:, :, :, :, :, 2]))
        mask = tf.minimum(mask, tf.abs(dot0[:, :, :, :, :, 0]))
        mask = tf.minimum(mask, tf.abs(dot1[:, :, :, :, :, 0]))
        mask = tf.where(mask > self.threshold, 1., mask)
        mask = tf.expand_dims(tf.expand_dims(mask, 5), 6)

        v = v * mask
        return v

    def get_config(self):
        config = super().get_config().copy()
        config['sigma'] = self.sigma
        config['threshold'] = self.threshold
        return config
