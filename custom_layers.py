import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class ProximalLayer(layers.Layer):
    """
    Single layer for an unrolled optimization network performing least squares minimization with a deep learning prior
    solved using the Proximal Gradient Algorithm.
    Uses only one contrast component obtained as an initial guess
    """

    def __init__(self, A, eta, prior):

        super(ProximalLayer, self).__init__()
        self.A = tf.Variable(initial_value=A, dtype=tf.float32, trainable=False)
        self.eta = tf.Variable(initial_value=eta, dtype=tf.float32, name="proxgrad_step", trainable=True)
        self.prior = prior

    def call(self, inputs, **kwargs):
        [xt, y] = inputs
        nabla_f = tf.matmul(tf.transpose(self.A), tf.matmul(self.A, xt) - y[0])
        zt1 = xt - (self.eta * nabla_f)
        zt1 = tf.reshape(zt1, (-1, 50, 50))
        zt1 = zt1[..., np.newaxis]
        xt1 = self.prior(zt1)
        xt1 = tf.reshape(xt1[:, :, :, 0], (-1, 2500, 1))
        return [xt1, y]


class QSProximalLayer(layers.Layer):
    """
    Single layer for an unrolled optimization network performing least squares minimization with a deep learning prior
    solved using the Proximal Gradient Algorithm.
    Uses only one contrast component obtained as an initial guess
    """

    def __init__(self, A, alpha, eta, prior):
        super(QSProximalLayer, self).__init__()
        self.A = tf.Variable(initial_value=A, dtype=tf.float32, trainable=False)
        self.alpha = tf.Variable(initial_value=alpha, dtype=tf.float32, trainable=True)
        self.eta = tf.Variable(initial_value=eta, dtype=tf.float32, name="proxgrad_step", trainable=True)
        self.prior = prior

    @staticmethod
    def difference_operator(m, num_grids, direction, sparse):

        d_row = np.zeros((1, num_grids))
        d_row[0, 0] = 1

        if direction == "horizontal":
            if sparse:
                d_row[0, 1] = -2
                d_row[0, 2] = 1
            else:
                d_row[0, 1] = -1
        elif direction == "vertical":
            if sparse:
                d_row[0, m] = -2
                d_row[0, 2 * m] = 1
            else:
                d_row[0, m] = -1
        else:
            raise ValueError("Invalid direction value for difference operator")

        rows = list()
        rows.append(d_row)
        for i in range(0, num_grids - 1):
            shifted_row = np.roll(d_row, 1)
            shifted_row[0, 0] = 0
            rows.append(shifted_row)
            d_row = shifted_row
        d = np.vstack([row for row in rows])

        return d

    def call(self, inputs, **kwargs):
        [xt, y] = inputs

        dim = self.A.shape[1]
        Dx = QSProximalLayer.difference_operator(50, dim, "horizontal", sparse=True)
        Dy = QSProximalLayer.difference_operator(50, dim, "vertical", sparse=True)
        qs = Dx.T @ Dx + Dy.T @ Dy
        qs_operator = tf.cast(qs, tf.float32)
        nabla_f = tf.matmul(tf.transpose(self.A), tf.matmul(self.A, xt) - y[0]) + self.alpha * tf.matmul(qs_operator, xt)
        zt1 = xt - (self.eta * nabla_f)

        zt1 = tf.reshape(zt1, (-1, 50, 50))
        zt1 = zt1[..., np.newaxis]
        xt1 = self.prior(zt1)

        xt1 = tf.reshape(xt1[:, :, :, 0], (-1, 2500, 1))
        return [xt1, y]


class PriorLayer(layers.Layer):

    def __init__(self, prior):

        super(PriorLayer, self).__init__()
        self.prior = prior

    def call(self, inputs, **kwargs):
        [xt, y] = inputs
        zt1 = tf.reshape(xt, (-1, 50, 50))
        zt1 = zt1[..., np.newaxis]
        xt1 = self.prior(zt1)
        xt1 = tf.reshape(xt1[:, :, :, 0], (-1, 2500, 1))
        return [xt1, y]


class UNetPriorLayer(layers.Layer):

    def __init__(self, prior):

        super(UNetPriorLayer, self).__init__()
        self.prior = prior

    def call(self, inputs, **kwargs):
        [xt] = inputs
        zt1 = tf.reshape(xt, (-1, 50, 50))
        zt1 = zt1[..., np.newaxis]
        xt1 = self.prior(zt1)
        xt1 = tf.reshape(xt1[:, :, :, 0], (-1, 2500, 1))
        return [xt1]


# class ComplexProximalLayer(layers.Layer):
#     """
#     Single layer for an unrolled optimization network performing least squares minimization with a deep learning prior
#     solved using the Proximal Gradient Algorithm.
#     Uses both real and imaginary parts of contrast obtained in the initial guess
#     """
#
#     def __init__(self, A, eta, prior):
#
#         super(NewProximalLayer, self).__init__()
#         self.A = tf.Variable(initial_value=A, dtype=tf.float32, trainable=False)
#         self.eta = tf.Variable(initial_value=eta, dtype=tf.float32, trainable=True)
#         self.prior = prior
#
#     def call(self, inputs, **kwargs):
#         [xr, xt, y] = inputs
#         x = tf.concat([xr, xt], axis=1)
#         nabla_f = tf.matmul(tf.transpose(self.A), tf.matmul(self.A, x) - y[0])
#         zt = x - (self.eta * nabla_f)
#
#         zt1 = zt[:, 2500:, :]
#         zt1 = tf.reshape(zt1, (-1, 50, 50))
#         zt1 = zt1[..., np.newaxis]
#         xr1 = tf.reshape(xr, (-1, 50, 50))
#         xr1 = xr1[..., np.newaxis]
#         ip = tf.concat([xr1, zt1], axis=3)
#         xt1 = self.prior(ip)
#
#         xt1 = tf.reshape(xt1[:, :, :, 0], (-1, 2500, 1))
#         return [xr, xt1, y]


class QSInitialGuessLayer(layers.Layer):
    """
    Layer to generate the initial guess using measurement data and quadratic smoothing regularizer
    Regularization parameter is a learnable parameter of the unrolled network.
    """

    def __init__(self, A, alpha, **kwargs):
        super().__init__(**kwargs)
        self.A = tf.Variable(initial_value=A, dtype=tf.float32, trainable=False)
        self.alpha = tf.Variable(initial_value=alpha, dtype=tf.float32, name="ig_reg_param", trainable=True)

    @staticmethod
    def difference_operator(m, num_grids, direction, sparse):

        d_row = np.zeros((1, num_grids))
        d_row[0, 0] = 1

        if direction == "horizontal":
            if sparse:
                d_row[0, 1] = -2
                d_row[0, 2] = 1
            else:
                d_row[0, 1] = -1
        elif direction == "vertical":
            if sparse:
                d_row[0, m] = -2
                d_row[0, 2 * m] = 1
            else:
                d_row[0, m] = -1
        else:
            raise ValueError("Invalid direction value for difference operator")

        rows = list()
        rows.append(d_row)
        for i in range(0, num_grids - 1):
            shifted_row = np.roll(d_row, 1)
            shifted_row[0, 0] = 0
            rows.append(shifted_row)
            d_row = shifted_row
        d = np.vstack([row for row in rows])

        return d

    def call(self, inputs, **kwargs):
        [data] = inputs
        dim = self.A.shape[1]
        Dx = QSInitialGuessLayer.difference_operator(50, dim, "horizontal", sparse=True)
        Dy = QSInitialGuessLayer.difference_operator(50, dim, "vertical", sparse=True)

        Dx = tf.cast(Dx, tf.float32)
        Dy = tf.cast(Dy, tf.float32)

        qs_operator = tf.matmul(tf.transpose(Dx), Dx) + tf.matmul(tf.transpose(Dy), Dy)
        inverse = tf.linalg.inv(tf.matmul(tf.transpose(self.A), self.A) + self.alpha * qs_operator)
        data_term = tf.matmul(tf.transpose(self.A), data)
        chi = tf.matmul(inverse, data_term)

        chi = tf.transpose(tf.reshape(chi, (-1, 50, 50)), perm=[0, 2, 1])
        chi = tf.reshape(chi, (-1, 2500, 1))

        chi = tf.nn.relu(chi)
        return [chi]


class InitialGuessLayer(layers.Layer):
    """
    Layer to generate the initial guess using measurement data and quadratic smoothing regularizer
    Regularization parameter is a learnable parameter of the unrolled network.
    """

    def __init__(self, A, **kwargs):
        super().__init__(**kwargs)
        self.A = tf.Variable(initial_value=A, dtype=tf.float32, trainable=False)

    def call(self, inputs, **kwargs):
        [data] = inputs

        inverse = tf.linalg.inv(tf.matmul(tf.transpose(self.A), self.A))
        data_term = tf.matmul(tf.transpose(self.A), data)
        chi = tf.matmul(inverse, data_term)

        chi = tf.transpose(tf.reshape(chi, (-1, 50, 50)), perm=[0, 2, 1])
        chi = tf.reshape(chi, (-1, 2500, 1))

        chi = tf.nn.relu(chi)
        return [chi]


class NNInitialGuessLayer(layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        [data] = inputs
        chi = layers.Dense(2500, activation='relu')(data[:, :, 0])
        return [chi]


class QSMultiInitialGuessLayer(layers.Layer):
    """
    Layer to generate the initial guess using measurement data and quadratic smoothing regularizer
    Regularization parameter is a learnable parameter of the unrolled network.
    """

    def __init__(self, A, alpha, **kwargs):
        super().__init__(**kwargs)
        self.A = tf.Variable(initial_value=A, dtype=tf.float32, trainable=False)
        self.alpha1 = tf.Variable(initial_value=1, dtype=tf.float32, name="ig_reg_param1", trainable=True)
        self.alpha2 = tf.Variable(initial_value=10, dtype=tf.float32, name="ig_reg_param2", trainable=True)

        self.w1 = tf.Variable(initial_value=0.5, dtype=tf.float32, name="ig_w1", trainable=True)
        self.w2 = tf.Variable(initial_value=0.5, dtype=tf.float32, name="ig_w2", trainable=True)

    @staticmethod
    def difference_operator(m, num_grids, direction, sparse):

        d_row = np.zeros((1, num_grids))
        d_row[0, 0] = 1

        if direction == "horizontal":
            if sparse:
                d_row[0, 1] = -2
                d_row[0, 2] = 1
            else:
                d_row[0, 1] = -1
        elif direction == "vertical":
            if sparse:
                d_row[0, m] = -2
                d_row[0, 2 * m] = 1
            else:
                d_row[0, m] = -1
        else:
            raise ValueError("Invalid direction value for difference operator")

        rows = list()
        rows.append(d_row)
        for i in range(0, num_grids - 1):
            shifted_row = np.roll(d_row, 1)
            shifted_row[0, 0] = 0
            rows.append(shifted_row)
            d_row = shifted_row
        d = np.vstack([row for row in rows])

        return d

    def call(self, inputs, **kwargs):
        [data] = inputs
        dim = self.A.shape[1]
        Dx = QSMultiInitialGuessLayer.difference_operator(50, dim, "horizontal", sparse=True)
        Dy = QSMultiInitialGuessLayer.difference_operator(50, dim, "vertical", sparse=True)

        Dx = tf.cast(Dx, tf.float32)
        Dy = tf.cast(Dy, tf.float32)

        qs_operator = tf.matmul(tf.transpose(Dx), Dx) + tf.matmul(tf.transpose(Dy), Dy)

        inverse = tf.linalg.inv(tf.matmul(tf.transpose(self.A), self.A) + self.alpha1 * qs_operator)
        data_term = tf.matmul(tf.transpose(self.A), data)
        chi1 = tf.matmul(inverse, data_term)
        chi1 = tf.transpose(tf.reshape(chi1, (-1, 50, 50)), perm=[0, 2, 1])
        chi1 = tf.reshape(chi1, (-1, 2500, 1))

        inverse = tf.linalg.inv(tf.matmul(tf.transpose(self.A), self.A) + self.alpha2 * qs_operator)
        data_term = tf.matmul(tf.transpose(self.A), data)
        chi2 = tf.matmul(inverse, data_term)
        chi2 = tf.transpose(tf.reshape(chi2, (-1, 50, 50)), perm=[0, 2, 1])
        chi2 = tf.reshape(chi2, (-1, 2500, 1))

        chi = self.w1 * chi1 + self.w2 * chi2

        chi = tf.nn.relu(chi)
        return [chi]


class QSRidgeInitialGuessLayer(layers.Layer):
    """
    Layer to generate the initial guess using measurement data and quadratic smoothing regularizer
    Regularization parameter is a learnable parameter of the unrolled network.
    """

    def __init__(self, A, alpha, **kwargs):
        super().__init__(**kwargs)
        self.A = tf.Variable(initial_value=A, dtype=tf.float32, trainable=False)
        self.alpha1 = tf.Variable(initial_value=alpha, dtype=tf.float32, name="ig_reg_param1", trainable=True)
        self.alpha2 = tf.Variable(initial_value=alpha, dtype=tf.float32, name="ig_reg_param2", trainable=True)

    @staticmethod
    def difference_operator(m, num_grids, direction, sparse):

        d_row = np.zeros((1, num_grids))
        d_row[0, 0] = 1

        if direction == "horizontal":
            if sparse:
                d_row[0, 1] = -2
                d_row[0, 2] = 1
            else:
                d_row[0, 1] = -1
        elif direction == "vertical":
            if sparse:
                d_row[0, m] = -2
                d_row[0, 2 * m] = 1
            else:
                d_row[0, m] = -1
        else:
            raise ValueError("Invalid direction value for difference operator")

        rows = list()
        rows.append(d_row)
        for i in range(0, num_grids - 1):
            shifted_row = np.roll(d_row, 1)
            shifted_row[0, 0] = 0
            rows.append(shifted_row)
            d_row = shifted_row
        d = np.vstack([row for row in rows])

        return d

    def call(self, inputs, **kwargs):
        [data] = inputs
        dim = self.A.shape[1]
        Dx = QSInitialGuessLayer.difference_operator(50, dim, "horizontal", sparse=True)
        Dy = QSInitialGuessLayer.difference_operator(50, dim, "vertical", sparse=True)

        Dx = tf.cast(Dx, tf.float32)
        Dy = tf.cast(Dy, tf.float32)

        qs_operator = tf.matmul(tf.transpose(Dx), Dx) + tf.matmul(tf.transpose(Dy), Dy)
        inverse = tf.linalg.inv(tf.matmul(tf.transpose(self.A), self.A) + self.alpha1 * qs_operator + self.alpha2 * tf.eye(dim))
        data_term = tf.matmul(tf.transpose(self.A), data)
        chi = tf.matmul(inverse, data_term)

        chi = tf.transpose(tf.reshape(chi, (-1, 50, 50)), perm=[0, 2, 1])
        chi = tf.reshape(chi, (-1, 2500, 1))

        chi = tf.nn.relu(chi)
        return [chi]


class ADMMTVLayer(layers.Layer):
    """
    Single layer for an unrolled optimization network performing least squares minimization with l1 constraints
    solved using the Alternating Direction Method of Multipliers Algorithm.
    """

    def __init__(self, A, alpha, eta, F):

        super(ADMMTVLayer, self).__init__()
        self.A = tf.Variable(initial_value=A, dtype=tf.float32, trainable=False)
        self.alpha = tf.Variable(initial_value=alpha, dtype=tf.float32, trainable=True)
        self.eta = tf.Variable(initial_value=eta, dtype=tf.float32, trainable=False)
        self.F = tf.constant(F, dtype=tf.float32)

    def soft_threshold(self, x, name=None):
        """ Proximal operator for the term ||x||_1"""
        with tf.name_scope(name or 'soft_threshold'):
            x = tf.convert_to_tensor(x, name='x', dtype=tf.float32)
            # threshold = self.alpha * self.eta
            # threshold = tf.convert_to_tensor(threshold, dtype=x.dtype, name='threshold')
            return tf.sign(x) * tf.maximum(tf.abs(x) - self.alpha*self.eta, 0.)

    def data_prox(self, x, y, name=None):
        with tf.name_scope(name or 'least_squares_prox'):
            self.A = tf.convert_to_tensor(self.A, name='A', dtype=tf.float32)
            x = tf.convert_to_tensor(x, name='x', dtype=tf.float32)
            y = tf.convert_to_tensor(y, name='y', dtype=tf.float32)
            # self.eta = tf.convert_to_tensor(self.eta, name='eta', dtype=tf.float32)

            # (eta A'A + F'F)^-1 * (eta A'b + F'(zt - ut))
            t1 = tf.linalg.inv(tf.matmul(tf.transpose(self.F), self.F) + self.eta * (tf.matmul(tf.transpose(self.A), self.A)))
            t2 = self.eta * tf.matmul(tf.transpose(self.A), y) + tf.matmul(tf.transpose(self.F), x)
            return tf.matmul(t1, t2)

    def call(self, inputs, **kwargs):
        [xt, zt, ut, y] = inputs
        xt1 = self.data_prox(zt - ut, y)
        thresh_matrix = tf.matmul(self.F, xt1) + ut
        thresh_matrix = tf.cast(thresh_matrix, dtype=tf.float32)
        zt1 = self.soft_threshold(thresh_matrix)
        ut1 = ut + tf.matmul(self.F, xt1) - zt1
        return [xt1, zt1, ut1, y]


