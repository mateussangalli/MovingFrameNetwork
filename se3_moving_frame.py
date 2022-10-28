import numpy as np
from scipy.special import eval_hermitenorm
from scipy.signal import correlate


def diff_center_of_mass(voxels):
    """" computes difference from the center of mass of each voxel within a volume of dimensions [B, H, W, D]"""
    x = np.linspace(0., 1., voxels.shape[1])
    y = np.linspace(0., 1., voxels.shape[2])
    z = np.linspace(0., 1., voxels.shape[3])
    xx, yy, zz = np.meshgrid(x, y, z)
    xx = xx[np.newaxis, :, :, :]
    yy = yy[np.newaxis, :, :, :]
    zz = zz[np.newaxis, :, :, :]
    center_x = np.sum(xx * voxels, (1, 2, 3)) / voxels.sum()
    center_y = np.sum(yy * voxels, (1, 2, 3)) / voxels.sum()
    center_z = np.sum(zz * voxels, (1, 2, 3)) / voxels.sum()
    center = np.stack([center_x, center_y, center_z], 1)
    pos = np.stack([xx, yy, zz], -1)
    diff = np.reshape(center, [-1, 1, 1, 1, 3]) - pos
    return diff


def depthconvx(inputs, kernel):
    shape = inputs.shape
    out = np.transpose(inputs, (1, 0, 2, 3))
    out = np.reshape(out, [shape[1], -1])
    out = correlate(out, kernel, 'same')
    out = np.reshape(out, [shape[1], shape[0], shape[2], shape[3]])
    out = np.transpose(out, (1, 0, 2, 3))
    return out


def depthconvy(inputs, kernel):
    shape = inputs.shape
    out = np.transpose(inputs, (2, 0, 1, 3))
    out = np.reshape(out, [shape[2], -1])
    out = correlate(out, kernel, 'same')
    out = np.reshape(out, [shape[2], shape[0], shape[1], shape[3]])
    out = np.transpose(out, (1, 2, 0, 3))
    return out


def depthconvz(inputs, kernel):
    shape = inputs.shape
    out = np.transpose(inputs, (3, 0, 1, 2))
    out = np.reshape(out, [shape[3], -1])
    out = correlate(out, kernel, 'same')
    out = np.reshape(out, [shape[3], shape[0], shape[1], shape[2]])
    out = np.transpose(out, (1, 2, 3, 0))
    return out


class SE3MovingFrame:
    def __init__(self, sigma, grad_sigmas=None, threshold=1e-5):
        self.threshold = threshold
        self.sigma = sigma
        width = int(np.round(4 * sigma))
        self.width = width
        if grad_sigmas is None:
            self.grad_sigmas = [sigma]
        else:
            self.grad_sigmas = grad_sigmas


        x = np.arange(-width, width + 1, dtype=np.float32)
        g0 = np.exp(-(x ** 2) / (2 * sigma ** 2))
        g0 /= np.sqrt(2 * np.pi * sigma)

        g = [g0]
        for n in range(1, 3):
            tmp = (1 - 2 * (n % 2)) * eval_hermitenorm(n, x / sigma) * g0
            tmp /= sigma ** n
            g.append(tmp)

        self.g = [p.reshape([-1, 1]) for p in g]

    def _get_derivatives_order1(self, inputs, sigma):
        width = int(np.round(4 * sigma))
        x = np.arange(-width, width + 1, dtype=np.float32)
        g0 = np.exp(-(x ** 2) / (2 * sigma ** 2))
        g0 /= np.sqrt(2 * np.pi * sigma)

        g = [g0]
        for n in range(1, 3):
            tmp = (1 - 2 * (n % 2)) * eval_hermitenorm(n, x / sigma) * g0
            tmp /= sigma ** n
            g.append(tmp)

        g = [p.reshape([-1, 1]) for p in g]

        ux = depthconvx(inputs, g[1])
        ux = depthconvy(ux, g[0])
        ux = depthconvz(ux, g[0])

        uy = depthconvx(inputs, g[0])
        uy = depthconvy(uy, g[1])
        uy = depthconvz(uy, g[0])

        uz = depthconvx(inputs, g[0])
        uz = depthconvy(uz, g[0])
        uz = depthconvz(uz, g[1])

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

    def __call__(self, inputs):
        uxx, uyy, uzz, uxy, uxz, uyz = self._get_derivatives_order2(inputs)
        grads = list()
        for sigma in self.grad_sigmas:
            ux, uy, uz = self._get_derivatives_order1(inputs, self.sigma)
            grad = np.stack([ux, uy, uz], -1)
            grads.append(grad)
        hess1 = np.stack([uxx, uxy, uxz], -1)
        hess2 = np.stack([uxy, uyy, uyz], -1)
        hess3 = np.stack([uxz, uyz, uzz], -1)
        hess = np.stack([hess1, hess2, hess3], -1)
        l, v = np.linalg.eigh(hess)
        l = l[..., ::-1]
        v = v[..., ::-1]


        v0 = v[..., 0]
        v1 = v[..., 1]
        v2 = v[..., 2]
        dot0 = 0
        dot1 = 0
        for grad in grads:
            tmp0 = np.sum(v0 * grad, -1, keepdims=True)
            tmp1 = np.sum(v1 * grad, -1, keepdims=True)
            dot0 = np.where(np.abs(tmp0) > np.abs(dot0), tmp0, dot0)
            dot1 = np.where(np.abs(tmp1) > np.abs(dot1), tmp1, dot1)


        v0 = np.where(dot0 < 0, -v0, v0)
        v1 = np.where(dot1 < 0, -v1, v1)
        v = np.stack([v0, v1, v2], -1)

        # v = np.where(np.sum(ori * v, -2, keepdims=True) < 0, -v, v)
        v[..., 2] = np.where(np.linalg.det(v)[..., np.newaxis] < 0, -v[..., 2], v[..., 2])
        # v = v * np.linalg.det(v)[..., np.newaxis, np.newaxis]
        mask = np.abs(l[:, :, :, :, 0] - l[:, :, :, :, 1])
        mask = np.minimum(mask, np.abs(l[:, :, :, :, 1] - l[:, :, :, :, 2]))
        # mask = np.minimum(mask, np.sum(grad ** 2, -1))
        mask = np.minimum(mask, np.abs(dot0[..., 0]))
        mask = np.minimum(mask, np.abs(dot1[..., 0]))
        # mask = np.minimum(mask, np.abs(np.sum(v[..., 0] * ori[..., 0], -1)))
        # mask = np.minimum(mask, np.abs(np.sum(v[..., 1] * ori[..., 0], -1)))
        mask = np.where(mask > self.threshold, 1., mask)[..., np.newaxis, np.newaxis]
        v = v * mask
        v = v[..., np.newaxis, :, :]
        return v

# class SE3MovingFrame:
#     def __init__(self, sigma, threshold=1e-7):
#         self.threshold = threshold
#         width = int(np.round(4 * sigma))
#
#         x = np.arange(-width, width + 1, dtype=np.float32)
#         g0 = np.exp(-(x ** 2) / (2 * sigma ** 2))
#         g0 /= np.sqrt(2 * np.pi * sigma)
#
#         g = [g0]
#         for n in range(1, 3):
#             tmp = (1 - 2 * (n % 2)) * eval_hermitenorm(n, x / sigma) * g0
#             tmp /= sigma ** n
#             g.append(tmp)
#
#         self.g = [p.reshape([-1, 1]) for p in g]
#
#     def _get_derivatives(self, inputs):
#         ux = depthconvx(inputs, self.g[1])
#         ux = depthconvy(ux, self.g[0])
#         ux = depthconvz(ux, self.g[0])
#
#         uy = depthconvx(inputs, self.g[0])
#         uy = depthconvy(uy, self.g[1])
#         uy = depthconvz(uy, self.g[0])
#
#         uz = depthconvx(inputs, self.g[0])
#         uz = depthconvy(uz, self.g[0])
#         uz = depthconvz(uz, self.g[1])
#
#         uxx = depthconvx(inputs, self.g[2])
#         uxx = depthconvy(uxx, self.g[0])
#         uxx = depthconvz(uxx, self.g[0])
#
#         uyy = depthconvx(inputs, self.g[0])
#         uyy = depthconvy(uyy, self.g[2])
#         uyy = depthconvz(uyy, self.g[0])
#
#         uzz = depthconvx(inputs, self.g[0])
#         uzz = depthconvy(uzz, self.g[0])
#         uzz = depthconvz(uzz, self.g[2])
#
#         uxy = depthconvx(inputs, self.g[1])
#         uxy = depthconvy(uxy, self.g[1])
#         uxy = depthconvz(uxy, self.g[0])
#
#         uxz = depthconvx(inputs, self.g[1])
#         uxz = depthconvy(uxz, self.g[0])
#         uxz = depthconvz(uxz, self.g[1])
#
#         uyz = depthconvx(inputs, self.g[0])
#         uyz = depthconvy(uyz, self.g[1])
#         uyz = depthconvz(uyz, self.g[1])
#         return ux, uy, uz, uxx, uyy, uzz, uxy, uxz, uyz
#
#     def __call__(self, inputs):
#         ux, uy, uz, uxx, uyy, uzz, uxy, uxz, uyz = self._get_derivatives(inputs)
#         C = ux / (np.sqrt(ux ** 2 + uy ** 2) + 1e-10)
#         S = uy / (np.sqrt(ux ** 2 + uy ** 2) + 1e-10)
#         R1 = np.zeros(inputs.shape + (3, 3))
#         R1[..., 0, 0] = C
#         R1[..., 1, 0] = -S
#         R1[..., 0, 1] = S
#         R1[..., 1, 1] = C
#         R1[..., 2, 2] = 1
#
#         ux2 = np.sqrt(ux ** 2 + uy ** 2)
#
#         C = ux2 / (np.sqrt(ux2 ** 2 + uz ** 2) + 1e-10)
#         S = uz / (np.sqrt(ux2 ** 2 + uz ** 2) + 1e-10)
#         R2 = np.zeros(inputs.shape + (3, 3))
#         R2[..., 0, 0] = C
#         R2[..., 2, 0] = -S
#         R2[..., 0, 2] = S
#         R2[..., 2, 2] = C
#         R2[..., 1, 1] = 1
#
#         R = np.matmul(R2, R1)
#
#         hess1 = np.stack([uxx, uxy, uxz], -1)
#         hess2 = np.stack([uxy, uyy, uyz], -1)
#         hess3 = np.stack([uxz, uyz, uzz], -1)
#         hess = np.stack([hess1, hess2, hess3], -1)
#         # hess_tr = np.matmul(R, np.matmul(hess, np.transpose(R, (0, 1, 2, 3, 5, 4))))
#         hess_tr = np.matmul(np.transpose(R, (0, 1, 2, 3, 5, 4)), np.matmul(hess, R))
#
#         A = hess_tr[..., 1:, 1:]
#         l, v = np.linalg.eigh(A)
#         R3 = np.zeros(inputs.shape + (3, 3))
#         # R3[..., 1:, 1:] = np.transpose(v, (0, 1, 2, 3, 5, 4))
#         R3[..., 1:, 1:] = v
#         R3[..., 0, 0] = 1
#         R = np.matmul(R3, R)
#         R = np.where((ux ** 2 + uy ** 2 + uz ** 2 < self.threshold)[..., np.newaxis, np.newaxis], 0., R)
#         R = np.transpose(R, (0,1,2,3,5,4))
#         return R[..., np.newaxis, :, :]
