import numpy as np
import tensorflow as tf
from scipy.special import eval_hermitenorm

EPS = 1e-5


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


@tf.function
def _order2_coef(frame, i, j):
    c = tf.expand_dims(frame[:, :, :, :, :, :, i], 5) * tf.expand_dims(frame[:, :, :, :, :, :, j], 6)
    c200 = c[:, :, :, :, :, 0, 0]
    c020 = c[:, :, :, :, :, 1, 1]
    c002 = c[:, :, :, :, :, 2, 2]
    c110 = c[:, :, :, :, :, 0, 1] + c[:, :, :, :, :, 1, 0]
    c101 = c[:, :, :, :, :, 0, 2] + c[:, :, :, :, :, 2, 0]
    c011 = c[:, :, :, :, :, 1, 2] + c[:, :, :, :, :, 2, 1]
    return tf.stack([c200, c110, c020, c101, c011, c002], -1)


@tf.function
def _order3_coef(frame, i, j, k):
    c = tf.expand_dims(tf.expand_dims(frame[:, :, :, :, :, :, i], 6), 7) \
        * tf.expand_dims(tf.expand_dims(frame[:, :, :, :, :, :, j], 5), 7) \
        * tf.expand_dims(tf.expand_dims(frame[:, :, :, :, :, :, k], 5), 5)
    c300 = c[:, :, :, :, :, 0, 0, 0]
    c030 = c[:, :, :, :, :, 1, 1, 1]
    c003 = c[:, :, :, :, :, 2, 2, 2]
    c210 = c[:, :, :, :, :, 1, 0, 0] + c[:, :, :, :, :, 0, 1, 0] + c[:, :, :, :, :, 0, 0, 1]
    c120 = c[:, :, :, :, :, 0, 1, 1] + c[:, :, :, :, :, 1, 0, 1] + c[:, :, :, :, :, 1, 1, 0]
    c201 = c[:, :, :, :, :, 2, 0, 0] + c[:, :, :, :, :, 0, 2, 0] + c[:, :, :, :, :, 0, 0, 2]
    c102 = c[:, :, :, :, :, 0, 2, 2] + c[:, :, :, :, :, 2, 0, 2] + c[:, :, :, :, :, 2, 2, 0]
    c021 = c[:, :, :, :, :, 2, 1, 1] + c[:, :, :, :, :, 1, 2, 1] + c[:, :, :, :, :, 1, 1, 2]
    c012 = c[:, :, :, :, :, 1, 2, 2] + c[:, :, :, :, :, 2, 1, 2] + c[:, :, :, :, :, 2, 2, 1]
    c111 = c[:, :, :, :, :, 0, 1, 2] + c[:, :, :, :, :, 0, 2, 1] + c[:, :, :, :, :, 1, 0, 2] \
           + c[:, :, :, :, :, 1, 2, 0] + c[:, :, :, :, :, 2, 0, 1] + c[:, :, :, :, :, 2, 1, 0]
    return tf.stack([c300, c210, c120, c030, c201, c111, c021, c102, c012, c003], -1)


# (3, 0, 0) (2, 1, 0) (1, 2, 0) (0, 3, 0) (2, 0, 1) (1, 1, 1) (0, 2, 1) (1, 0, 2) (0, 1, 2) (0, 0, 3)


class Order2Coefficients(tf.keras.layers.Layer):
    def call(self, frame):
        coefs = list()
        for i in range(3):
            for j in range(i, 3):
                coefs.append(_order2_coef(frame, i, j))
        return tf.stack(coefs, -1)


class Order3Coefficients(tf.keras.layers.Layer):
    def call(self, frame):
        coefs = list()
        for i in range(3):
            for j in range(i, 3):
                for k in range(j, 3):
                    coefs.append(_order3_coef(frame, i, j, k))
        return tf.stack(coefs, -1)


class SE3MovingFrameInvariants(tf.keras.layers.Layer):
    def __init__(self, sigma, width, order=2,
                 **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.width = width
        self.order = int(order)
        # assuming channels last
        self.channel_axis = -1

        x = np.arange(-width, width + 1, dtype=np.float32)
        g0 = np.exp(-(x ** 2) / (2 * sigma ** 2))
        g0 /= np.sqrt(2 * np.pi * sigma)

        g = [g0]
        for n in range(1, min(4, self.order + 1)):
            tmp = (1 - 2 * (n % 2)) * eval_hermitenorm(n, x / sigma) * g0
            tmp /= sigma ** n
            g.append(tmp)

        self.g = [p.reshape([-1, 1, 1, 1]) for p in g]

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        n_channels = input_shape[0][-1]
        self.n_channels = n_channels
        self.g = [np.concatenate(n_channels * [p], 2) for p in self.g]
        self.g = [tf.constant(p, dtype=tf.float32) for p in self.g]

        super().build(input_shape)

    def _get_derivatives(self, inputs):
        tmp = depthconvx(inputs, self.g[0])
        tmp = depthconvy(tmp, self.g[0])
        tmp = depthconvz(tmp, self.g[0])
        out = [tmp]
        for n in range(1, min(4, self.order + 1)):
            for i in range(0, n + 1):
                for j in range(i, n + 1):
                    tmp = depthconvx(inputs, self.g[n - j])
                    tmp = depthconvy(tmp, self.g[j - i])
                    tmp = depthconvz(tmp, self.g[i])
                    out.append(tmp)
        return tf.stack(out, -1)

    def call(self, inputs):
        if self.order == 2:
            f, frame, coefs_order2 = inputs
        elif self.order > 2:
            f, frame, coefs_order2, coefs_order3 = inputs
        shape = tf.shape(f)
        derivatives = self._get_derivatives(f)
        u = derivatives[:, :, :, :, :, 0]

        # return tf.concat([u, ux, uy, uz, uxx, uxy, uyy, uxz, uyz, uzz], -1)
        # grad = tf.stack([ux, uy, uz], -1)
        grad = derivatives[:, :, :, :, :, 1:4]
        grad_tr = tf.expand_dims(grad, 6)
        grad_tr = tf.reduce_sum(grad_tr * frame, 5)
        grad_tr = tf.reshape(grad_tr, [shape[0], shape[1], shape[2], shape[3], self.n_channels * 3])

        vec_order2 = derivatives[:, :, :, :, :, 4:10]
        vec_order2 = tf.reduce_sum(coefs_order2 * tf.expand_dims(vec_order2, 6), 5)
        vec_order2 = tf.reshape(vec_order2, [shape[0], shape[1], shape[2], shape[3], self.n_channels * 6])
        if self.order > 2:
            vec_order3 = derivatives[:, :, :, :, :, 10:]
            vec_order3 = tf.reduce_sum(coefs_order3 * tf.expand_dims(vec_order3, 6), 5)
            vec_order3 = tf.reshape(vec_order3, [shape[0], shape[1], shape[2], shape[3], self.n_channels * 10])
            return tf.concat([f, grad_tr, vec_order2, vec_order3], -1)
        else:
            return tf.concat([u, grad_tr, vec_order2], -1)

        # uxx = derivatives[:, :, :, :, :, 4]
        # uxy = derivatives[:, :, :, :, :, 5]
        # uyy = derivatives[:, :, :, :, :, 6]
        # uxz = derivatives[:, :, :, :, :, 7]
        # uyz = derivatives[:, :, :, :, :, 8]
        # uzz = derivatives[:, :, :, :, :, 9]
        # hess1 = tf.stack([uxx, uxy, uxz], -1)
        # hess2 = tf.stack([uxy, uyy, uyz], -1)
        # hess3 = tf.stack([uxz, uyz, uzz], -1)
        # hess = tf.stack([hess1, hess2, hess3], -1)

        # hess_tr = tf.matmul(frame, tf.matmul(hess, frame), transpose_a=True)
        # hess_tr1 = tf.reshape(hess_tr[:, :, :, :, :, :, 0],
        #                       [shape[0], shape[1], shape[2], shape[3], self.n_channels * 3])
        # hess_tr2 = tf.reshape(hess_tr[:, :, :, :, :, 1:, 1],
        #                       [shape[0], shape[1], shape[2], shape[3], self.n_channels * 2])
        # hess_tr3 = hess_tr[:, :, :, :, :, 2, 2]
        # invs_order2 = tf.concat([hess_tr1, hess_tr2, hess_tr3], -1)
        # tf.print(tf.reduce_mean(invs_order2, [0, 1, 2, 3]))
        # tf.print(tf.reduce_mean(vec_order2, [0, 1, 2, 3]))
        # return tf.concat([f, grad_tr, hess_tr1, hess_tr2, hess_tr3], -1)
        # return tf.concat([f, grad_tr], -1)
        # return tf.concat([hess_tr1, hess_tr2, hess_tr3], -1)

    def get_config(self):
        config = super().get_config().copy()
        config['sigma'] = self.sigma
        config['width'] = self.width
        config['order'] = self.order
        return config
