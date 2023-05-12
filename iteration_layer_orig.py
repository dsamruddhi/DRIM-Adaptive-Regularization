import numpy as np
from scipy.special import jv as bessel1
from scipy.special import hankel1

import tensorflow as tf
from tensorflow.keras import layers

from config import *


class DRIMLayer(layers.Layer):

    def __init__(self,
                 alpha,
                 m,
                 grid_positions,
                 direct_field,
                 incident_field,
                 G_freespace,
                 G_freespace_scaled):

        super(DRIMLayer, self).__init__()
        self.alpha = tf.Variable(initial_value=alpha, dtype=tf.float64, name="reg_param", trainable=True)
        self.m = m

        self.grid_positions = grid_positions

        self.direct_field = direct_field
        self.incident_field = incident_field
        self.G_freespace = G_freespace
        self.G_freespace_scaled = G_freespace_scaled

        self.num_grids = self.m ** 2
        self.grid_length = doi_length / self.m
        self.grid_radius = np.sqrt(self.grid_length ** 2 / np.pi)
        self.grid_area = (4 * np.pi * self.grid_radius / (2 * wave_number)) * bessel1(1, wave_number * self.grid_radius)

        self.wave_number = tf.cast(wave_number, tf.float32)
        self.impedance = tf.cast(impedance, tf.float32)

        self.C1 = -self.impedance * np.pi * (self.grid_radius / 2)
        self.C1 = tf.cast(self.C1, tf.complex64)

        self.C2 = bessel1(1, self.wave_number * self.grid_radius)
        self.C3 = hankel1(1, self.wave_number * self.grid_radius)

        self.Z = tf.Variable(1 + 1j, dtype=tf.complex64, shape=tf.TensorShape(None), name='Z', trainable=False)
        self.z1 = tf.Variable(tf.zeros((m**2, 1), dtype=tf.complex64), name='z1', trainable=False)
        self.induced_current = tf.Variable(tf.zeros((self.m ** 2, tx_count), dtype=tf.complex64), name='induced_current', trainable=False)

        self.incident_field_iter = tf.Variable(self.incident_field, dtype=tf.complex64, name='layer_incident_field', trainable=False)
        self.G_iter = tf.Variable(self.G_freespace, dtype=tf.complex64, name='layer_greens_function', trainable=False)

        self.A = tf.Variable(tf.zeros((len(sensor_links), self.num_grids), dtype=tf.complex64), name='layer_model', trainable=False)

    def call(self, inputs, **kwargs):

        [epsilon_r_iter, chi_iter, total_power] = inputs

        # forward code

        """" 8. Forward: Grids containing object """

        unrolled_scatterer = tf.reshape(tf.transpose(epsilon_r_iter), (self.m ** 2, 1))
        object_grids = tf.gather(tf.where(unrolled_scatterer != 1), [0], axis=1)

        """" 9. Forward: Method of Moment """

        unroll_x = self.grid_positions[0].reshape(self.grid_positions[0].size, order='F')
        unroll_y = self.grid_positions[1].reshape(self.grid_positions[1].size, order='F')

        unroll_x = tf.cast(unroll_x, tf.float32)
        unroll_y = tf.cast(unroll_y, tf.float32)

        obj_x = tf.gather(unroll_x, object_grids)
        obj_y = tf.gather(unroll_y, object_grids)

        s = tf.shape(object_grids)

        unrolled_scatterer = tf.cast(unrolled_scatterer, tf.float32)

        self.Z.assign(tf.zeros((s[0], s[0]), dtype=tf.complex64))

        for index in tf.range(s[0]):
            value = object_grids[index]
            x_incident = obj_x[index]
            y_incident = obj_y[index]

            dipole_distances = tf.sqrt((x_incident - obj_x) ** 2 + (y_incident - obj_y) ** 2)
            dipole_distances = tf.cast(dipole_distances, tf.float32)

            # a1 = hankel1(0, tf.math.scalar_mul(self.wave_number, dipole_distances))
            a1 = tf.compat.v1.py_function(func=hankel1, inp=[0, tf.math.scalar_mul(self.wave_number, dipole_distances)], Tout=tf.complex64)
            b1 = (self.impedance * unrolled_scatterer[value[0]]) / (self.wave_number * (unrolled_scatterer[value[0]] - 1))
            b1 = tf.complex(0.0, b1)

            self.z1.assign(self.C1 * self.C2 * a1)
            self.z1[index].assign(self.C1 * self.C3 - b1)

            self.Z[index, :].assign(self.z1[:, 0])

        """" 10. Forward: Induced current due to object with epsilon_r, non-freespace incident field, 
        inhomogeneous Green's function """

        field_on_object = tf.gather(-self.incident_field, object_grids)
        field_on_object = tf.squeeze(field_on_object)
        field_on_object = tf.cast(field_on_object, tf.complex64)

        G_object_to_rx = tf.gather(-self.G_freespace, object_grids)
        G_object_to_rx = tf.squeeze(G_object_to_rx)
        G_object_to_rx = tf.cast(G_object_to_rx, tf.complex64)

        # induced current due to tx for estimating total field [tx DoI interaction]
        J1 = tf.matmul(tf.linalg.inv(self.Z), field_on_object)
        # induced current due to rx as pseudo source for estimating inhomogenous Green's function [DoI rx interaction]
        G_J1 = tf.matmul(tf.linalg.inv(self.Z), G_object_to_rx)

        for i in range(len(object_grids)):
            self.induced_current[object_grids[i, 0], :].assign(J1[i, :])

            # Quantities used during inverse
            l = 1j
            l = tf.cast(l, tf.complex64)

            c = (unrolled_scatterer[object_grids[i, 0]] - 1) * wave_number
            c = tf.cast(c, tf.complex64)

            self.incident_field_iter[object_grids[i, 0], :].assign(tf.math.divide(tf.math.multiply(l, impedance * J1[object_grids[i, 0], :]), c))
            self.G_iter[object_grids[i, 0], :].assign(tf.math.divide(tf.math.multiply(l, impedance * G_J1[object_grids[i, 0], :]), c))

        G_iter = tf.transpose(self.G_iter)

        """" 11. Forward: Scattered field collected at the receivers hen object with epsilon_r is kept in the DoI"""

        scattered_field_iter = self.G_freespace_scaled @ self.induced_current

        """" 12. Forward: Total field at receiver when object with epsilon_r is kept in the DoI"""

        def remove_nan(field):
            indices = [x * (tx_count + 1) for x in range(0, tx_count)]
            field = tf.reshape(tf.transpose(field), (tx_count * rx_count, 1))
            a_vecs = tf.unstack(field)
            for i in reversed(indices):
                del a_vecs[i]
            a_new = tf.stack(a_vecs)
            k = tf.transpose(tf.reshape(a_new, (tx_count, rx_count - 1)))
            return k

        total_field_iter = self.direct_field + scattered_field_iter

        # inverse code

        """" 13. Inverse: Rytov model """

        for i, pair in enumerate(sensor_links):
            gg = self.grid_area * (wave_number ** 2) * tf.multiply(G_iter[pair[1], :], tf.transpose(self.incident_field_iter[:, pair[0]]))/ total_field_iter[pair[1], pair[0]]
            self.A[i, :].assign(gg)
        A_real = tf.math.real(self.A)
        A_imag = tf.math.imag(self.A)

        H_iter = tf.concat((A_real, -A_imag), axis=1)
        H_iter = tf.cast(H_iter, tf.float64)

        """" 14. Forward/Inverse: RSS values at receiver """

        total_field_iter = remove_nan(total_field_iter)

        def log10(num):
            numerator = tf.math.log(num)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

        def get_power(field):
            power = ((tf.abs(field) + noise_level) ** 2) * (wavelength ** 2) / (4 * np.pi * impedance)
            power = 10 * log10(power / 1e-3)
            return power

        total_power_iter = get_power(total_field_iter)

        """" 14. Inverse: Rytov data """

        a = (total_power - total_power_iter)
        a = tf.cast(a, tf.float64)
        data_iter = a / (20 * log10(np.exp(1)))

        data_iter = tf.reshape(tf.transpose(data_iter), (tx_count*(rx_count-1), 1))

        """" 15. Inverse: Ridge regression """

        dim = H_iter.shape[1]
        lambda_max = tf.linalg.norm(tf.transpose(H_iter) @ data_iter, 2)
        chi = tf.matmul(tf.linalg.inv(tf.matmul(tf.transpose(H_iter), H_iter) + lambda_max * self.alpha * tf.eye(dim, dtype=tf.float64)), tf.matmul(tf.transpose(H_iter), data_iter))

        """" 16. Inverse: Reshape chi """

        delta_chi_r = chi[:self.m ** 2]
        delta_chi_r = tf.transpose(tf.reshape(delta_chi_r, (self.m, self.m)))

        delta_chi_i = chi[self.m ** 2:]
        delta_chi_i = tf.transpose(tf.reshape(delta_chi_i, (self.m, self.m)))

        delta_chi_r = tf.cast(delta_chi_r, tf.float32)
        delta_chi_i = tf.cast(delta_chi_i, tf.float32)

        delta_chi = tf.complex(delta_chi_r, delta_chi_i)
        chi_iter = chi_iter + delta_chi
        epsilon_r_iter = chi_iter + 1

        return [epsilon_r_iter, chi_iter]
