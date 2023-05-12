from scipy.special import jv as bessel1
from scipy.special import hankel1

import tensorflow as tf
from tensorflow.keras import layers

from config import *


class InitializationLayer(layers.Layer):

    def __init__(self,
                 alpha,
                 m,
                 grid_positions,
                 direct_field,
                 direct_power,
                 incident_field,
                 distance):

        super(InitializationLayer, self).__init__()
        self.alpha = tf.Variable(initial_value=alpha, dtype=tf.float32, name="reg_param", trainable=True)

        self.m = m
        self.grid_positions = grid_positions

        self.direct_field = direct_field
        self.direct_power = direct_power
        self.incident_field = incident_field

        self.distance = distance

        self.num_grids = self.m ** 2
        self.grid_length = doi_length / self.m
        self.grid_radius = np.sqrt(self.grid_length ** 2 / np.pi)

    def call(self, inputs, **kwargs):

        [total_power] = inputs

        A = np.zeros((num_links, self.num_grids), dtype=complex)

        G_init = (1j * np.pi * self.grid_radius / (2 * wave_number)) * \
                 bessel1(1, wave_number * self.grid_radius) * hankel1(0, wave_number * np.transpose(self.distance))

        for i, pair in enumerate(sensor_links):
            A[i, :] = (wave_number ** 2) * np.divide(
                np.multiply(G_init[pair[1], :], np.transpose(self.incident_field[:, pair[0]])),
                self.direct_field[pair[1], pair[0]])

        A_real = np.real(A)
        A_imag = np.imag(A)
        H_init = np.concatenate((A_real, -A_imag), axis=1)

        H_init = tf.cast(H_init, tf.float32)

        """" 8. Initialization: Get Rytov data """

        data_init = (total_power - self.direct_power) / (20 * np.log10(np.exp(1)))
        data_init = tf.reshape(tf.transpose(data_init), (num_links, 1))

        data_init = tf.cast(data_init, tf.float32)

        """" 9. Initialization: Ridge regression """

        dim = H_init.shape[1]
        lambda_max = tf.linalg.norm(tf.matmul(tf.transpose(H_init), data_init), 2)
        chi = tf.linalg.inv((tf.transpose(H_init) @ H_init) + lambda_max * self.alpha * tf.eye(dim)) @ tf.transpose(H_init) @ data_init

        """" 16. Initialization: Reshape chi """

        chi_r = chi[:self.m ** 2]
        chi_r = tf.transpose(tf.reshape(chi_r, (self.m, self.m)))

        chi_i = chi[self.m ** 2:]
        chi_i = tf.transpose(tf.reshape(chi_i, (self.m, self.m)))

        chi_init = tf.complex(chi_r, chi_i)
        epsilon_r_init = chi_init + 1

        return [epsilon_r_init, chi_init]
