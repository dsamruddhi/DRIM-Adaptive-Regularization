from config import *

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.special import jv as bessel1
from scipy.special import hankel1
from scipy.io import loadmat, savemat

if __name__ == '__main__':

    """" 1. DRIM config parameters """

    m = 20
    grid_length = doi_length / m
    num_grids = m ** 2

    # """ Grid positions/ centroids """
    centroids_x = np.arange(start=- doi_length / 2 + grid_length / 2, stop=doi_length / 2, step=grid_length)
    centroids_y = np.arange(start=doi_length / 2 - grid_length / 2, stop=-doi_length / 2, step=-grid_length)
    grid_positions = np.meshgrid(centroids_x, centroids_y)

    # Grid radius
    grid_radius = np.sqrt(grid_length ** 2 / np.pi)

    # Grid area
    grid_area = (4 * np.pi * grid_radius / (2 * wave_number)) * bessel1(1, wave_number * grid_radius)

    # """ Constants used in code """
    C1 = -impedance * np.pi * (grid_radius / 2)
    C2 = bessel1(1, wave_number * grid_radius)
    C3 = hankel1(1, wave_number * grid_radius)

    # Other parameters
    iterations = 1
    reg_param = 0.01

    # variables used in code
    total_field_re = []
    epsilon_re = []

    """" 2. DoI permittivity profile - Ground truth to be used for comparison """

    # size = 0.015
    # permittivity = 3.3
    # center_x = 0
    # center_y = 0
    #
    # epsilon_r_GT = np.ones((m, m), dtype=complex)
    #
    # # Circle
    # # epsilon_r_GT[(grid_positions[0] - -0.005) ** 2 + (grid_positions[1] - 0.045) ** 2 <= size ** 2] = permittivity
    # # epsilon_r_GT[(grid_positions[0] - -0.012) ** 2 + (grid_positions[1] + 0.045) ** 2 <= size ** 2] = permittivity
    #
    # mask = ((grid_positions[0] <= center_x + 0.04) & (grid_positions[0] >= center_x -0.04) &
    #         (grid_positions[1] <= center_y + 0.015) & (grid_positions[1] >= center_y - 0.015))
    # epsilon_r_GT[mask] = permittivity

    epsilon_r_GT = loadmat(r"C:\Users\dsamr\OneDrive - HKUST Connect\MPHIL RESEARCH\DATA\DRIM_ADAPTIVE_3\scatterer_data_inverse_20\90.mat")['scatterer']

    # Plot
    plt.figure(1)
    plt.imshow(epsilon_r_GT, cmap=plt.cm.jet, extent=[-doi_length / 2, doi_length / 2, -doi_length / 2, doi_length / 2])
    plt.colorbar()
    plt.show()

    """" Epsilon_r and Chi """

    epsilon_r_iter = np.zeros((m, m), dtype=complex)
    chi = np.zeros((m, m), dtype=complex)

    """" 3. Load measurement data """

    direct_power = loadmat(r"C:\Users\dsamr\OneDrive - HKUST Connect\MPHIL RESEARCH\DATA\DRIM_ADAPTIVE_3\measurement_empty\direct_power.mat")["direct_power"]
    total_power = loadmat(r"C:\Users\dsamr\OneDrive - HKUST Connect\MPHIL RESEARCH\DATA\DRIM_ADAPTIVE_3\measurement_scatterer\0.mat")["total_power"]

    total_field = loadmat(r"C:\Users\dsamr\OneDrive - HKUST Connect\MPHIL RESEARCH\DATA\DRIM_ADAPTIVE_3\measurement_scatterer\0.mat")["total_field"]

    """" 4. Direct field from transmitter to receiver in free space """

    tx_xcoord = [pos[0] for pos in sensor_positions]
    tx_ycoord = [pos[1] for pos in sensor_positions]

    rx_xcoord = [pos[0] for pos in sensor_positions]
    rx_ycoord = [pos[1] for pos in sensor_positions]

    [xtd, xrd] = np.meshgrid(tx_xcoord, rx_xcoord)
    [ytd, yrd] = np.meshgrid(tx_ycoord, rx_ycoord)
    dist = np.sqrt((xtd - xrd) ** 2 + (ytd - yrd) ** 2)
    direct_field = (1j / 4) * hankel1(0, wave_number * dist)

    """" 5. Incident field from transmitter on all DoI grids """

    grid_xcoord = grid_positions[0]
    grid_xcoord = grid_xcoord.reshape(grid_xcoord.size, order='F')

    grid_ycoord = grid_positions[1]
    grid_ycoord = grid_ycoord.reshape(grid_ycoord.size, order='F')

    [xti, xsi] = np.meshgrid(tx_xcoord, grid_xcoord)
    [yti, ysi] = np.meshgrid(tx_ycoord, grid_ycoord)

    dist = np.sqrt((xti - xsi) ** 2 + (yti - ysi) ** 2)
    incident_field = (1j / 4) * hankel1(0, wave_number * dist)

    """" 6. Free space Green's function & scaled by a constant """

    [xts, xss] = np.meshgrid(tx_xcoord, grid_xcoord)
    [yts, yss] = np.meshgrid(tx_ycoord, grid_ycoord)

    dist = np.sqrt((xts - xss)**2 + (yts - yss)**2)

    G_freespace = (1j / 4) * hankel1(0, wave_number * dist)
    G_freespace_scaled = -impedance * np.pi * (grid_radius / 2) * bessel1(1, wave_number * grid_radius) * hankel1(0, wave_number * np.transpose(dist))

    """" 7. Initialization: Get Rytov model """

    A = np.zeros((len(sensor_links), num_grids), dtype=complex)
    G_init = (1j * np.pi * grid_radius / (2 * wave_number)) * \
            bessel1(1, wave_number * grid_radius) * hankel1(0, wave_number * np.transpose(dist))

    for i, pair in enumerate(sensor_links):
        A[i, :] = (wave_number ** 2) * np.divide(np.multiply(G_init[pair[1], :], np.transpose(incident_field[:, pair[0]])),
                                                 direct_field[pair[1], pair[0]])

    A_real = np.real(A)
    A_imag = np.imag(A)
    H_init = np.concatenate((A_real, -A_imag), axis=1)
    H_init = tf.cast(H_init, tf.float32)

    """" 8. Initialization: Get Rytov data """

    data_init = (total_power - direct_power) / (20 * np.log10(np.exp(1)))
    # data_init = data_init.reshape(data_init.size, order='F')
    data_init = tf.reshape(tf.transpose(data_init), (num_links, 1))
    data_init = tf.cast(data_init, tf.float32)

    """" 9. Initialization: Ridge regression """

    dim = H_init.shape[1]
    lambda_max = np.linalg.norm(np.transpose(H_init) @ data_init, 2)
    chi_init = tf.linalg.inv((tf.transpose(H_init) @ H_init) + lambda_max * reg_param * tf.eye(dim)) @ tf.transpose(
        H_init) @ data_init

    """" 16. Initialization: Reshape chi """

    chi_r = chi_init[:m ** 2]
    chi_r = np.reshape(chi_r, (m, m), order='F')

    chi_i = chi_init[m ** 2:]
    chi_i = np.reshape(chi_i, (m, m), order='F')

    chi_init = chi_r + 1j * chi_i
    epsilon_r_init = chi_init + 1

    """" 7. Loop - { [forward, inverse] - [forward, inverse] ... } """

    epsilon_r_iter = epsilon_r_init
    chi_iter = chi_init

    for iteration in range(0, iterations):

        """" TF variables """

        Z = tf.Variable(1 + 1j, dtype=tf.complex64, shape=tf.TensorShape(None), name='Z', trainable=True)
        z1 = tf.Variable(tf.zeros((m ** 2, 1), dtype=tf.complex64), name='z1', trainable=True)
        induced_current = tf.Variable(tf.zeros((m ** 2, tx_count), dtype=tf.complex64), name='induced_current')

        incident_field_iter = tf.Variable(incident_field, dtype=tf.complex64, name='layer_incident_field',
                                          trainable=True)
        G_iter = tf.Variable(G_freespace, dtype=tf.complex64, name='layer_greens_function', trainable=True)

        A = tf.Variable(tf.zeros((len(sensor_links), num_grids), dtype=tf.complex64), name='layer_model')

        # forward code

        """" 8. Forward: Grids containing object """

        unrolled_scatterer = tf.reshape(tf.transpose(epsilon_r_iter), (m ** 2, 1))
        object_grids = tf.gather(tf.where(unrolled_scatterer != 1), [0], axis=1)

        """" 9. Forward: Method of Moment """

        unroll_x = grid_positions[0].reshape(grid_positions[0].size, order='F')
        unroll_y = grid_positions[1].reshape(grid_positions[1].size, order='F')

        unroll_x = tf.cast(unroll_x, tf.float32)
        unroll_y = tf.cast(unroll_y, tf.float32)

        obj_x = tf.gather(unroll_x, object_grids)
        obj_y = tf.gather(unroll_y, object_grids)

        s = tf.shape(object_grids)

        unrolled_scatterer = tf.cast(unrolled_scatterer, tf.complex64)

        Z.assign(tf.zeros((s[0], s[0]), dtype=tf.complex64))

        for index in tf.range(s[0]):
            value = object_grids[index]
            x_incident = obj_x[index]
            y_incident = obj_y[index]

            dipole_distances = tf.sqrt((x_incident - obj_x) ** 2 + (y_incident - obj_y) ** 2)
            dipole_distances = tf.cast(dipole_distances, tf.float32)

            # a1 = hankel1(0, tf.math.scalar_mul(self.wave_number, dipole_distances))
            a1 = tf.compat.v1.py_function(func=hankel1, inp=[0, tf.math.scalar_mul(wave_number, dipole_distances)],
                                          Tout=tf.complex64)
            b1 = (impedance * unrolled_scatterer[value[0]]) / (
                        wave_number * (unrolled_scatterer[value[0]] - 1))

            z1.assign(C1 * C2 * a1)
            z1[index].assign(C1 * C3 - 1j * b1)

            Z[index, :].assign(z1[:, 0])

        """" 10. Forward: Induced current due to object with epsilon_r, non-freespace incident field, 
        inhomogeneous Green's function """

        field_on_object = tf.gather(-incident_field, object_grids)
        field_on_object = tf.squeeze(field_on_object)
        field_on_object = tf.cast(field_on_object, tf.complex64)

        G_object_to_rx = tf.gather(-G_freespace, object_grids)
        G_object_to_rx = tf.squeeze(G_object_to_rx)
        G_object_to_rx = tf.cast(G_object_to_rx, tf.complex64)

        # induced current due to tx for estimating total field [tx DoI interaction]
        J1 = tf.matmul(tf.linalg.inv(Z), field_on_object)
        # induced current due to rx as pseudo source for estimating inhomogenous Green's function [DoI rx interaction]
        G_J1 = tf.matmul(tf.linalg.inv(Z), G_object_to_rx)

        for i in range(len(object_grids)):
            induced_current[object_grids[i, 0], :].assign(J1[i, :])

            # Quantities used during inverse
            l = 1j
            l = tf.cast(l, tf.complex64)

            c = (unrolled_scatterer[object_grids[i, 0]] - 1) * wave_number
            c = tf.cast(c, tf.complex64)

            incident_field_iter[object_grids[i, 0], :].assign(
                tf.math.divide(tf.math.multiply(l, impedance * J1[object_grids[i, 0], :]), c))
            G_iter[object_grids[i, 0], :].assign(
                tf.math.divide(tf.math.multiply(l, impedance * G_J1[object_grids[i, 0], :]), c))

        G_iter = tf.transpose(G_iter)

        """" 11. Forward: Scattered field collected at the receivers hen object with epsilon_r is kept in the DoI"""

        scattered_field_iter = G_freespace_scaled @ induced_current

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

        total_field_iter = direct_field + scattered_field_iter

        # inverse code

        """" 13. Inverse: Rytov model """

        for i, pair in enumerate(sensor_links):
            gg = grid_area * (wave_number ** 2) * tf.multiply(G_iter[pair[1], :],
                                                                   tf.transpose(incident_field_iter[:, pair[0]])) / \
                 total_field_iter[pair[1], pair[0]]
            A[i, :].assign(gg)
        A_real = tf.math.real(A)
        A_imag = tf.math.imag(A)

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

        data_iter = tf.reshape(tf.transpose(data_iter), (tx_count * (rx_count - 1), 1))

        """" 15. Inverse: Ridge regression """

        dim = H_iter.shape[1]
        lambda_max = tf.linalg.norm(tf.transpose(H_iter) @ data_iter, 2)
        chi = tf.matmul(
            tf.linalg.inv(tf.matmul(tf.transpose(H_iter), H_iter) + lambda_max * reg_param * tf.eye(dim, dtype=tf.float64)),
            tf.matmul(tf.transpose(H_iter), data_iter))

        """" 16. Inverse: Reshape chi """

        delta_chi_r = chi[:m ** 2]
        delta_chi_r = tf.transpose(tf.reshape(delta_chi_r, (m, m)))

        delta_chi_i = chi[m ** 2:]
        delta_chi_i = tf.transpose(tf.reshape(delta_chi_i, (m, m)))

        delta_chi_r = tf.cast(delta_chi_r, tf.float64)
        delta_chi_i = tf.cast(delta_chi_i, tf.float64)

        delta_chi = tf.complex(delta_chi_r, delta_chi_i)
        chi_iter = chi_iter + delta_chi
        epsilon_r_iter = chi_iter + 1

        """" 17. Evaluation criteria - relative errors for total field and epsilon_r """

        epr = tf.identity(epsilon_r_iter)

        # Relative error for total field
        total_field_err = tf.linalg.norm(tf.cast(tf.math.abs(tf.reshape(total_field, tf.size(total_field))), tf.float64)
                                         - tf.cast(tf.math.abs(tf.reshape(total_field_iter, tf.size(total_field_iter))), tf.float64), 1) \
                          / tf.linalg.norm(tf.cast(tf.math.abs(tf.reshape(total_field, tf.size(total_field))), tf.float64), 1)

        # Relative error for epsilon_r
        epsilon_err = tf.norm(tf.cast(tf.math.abs(tf.reshape(epsilon_r_GT, tf.size(epsilon_r_GT))), tf.float64)
                                     - tf.cast(tf.math.abs(tf.reshape(epr, tf.size(epr))), tf.float64), 2) \
                      / tf.norm(tf.cast(tf.math.abs(tf.reshape(epsilon_r_GT, tf.size(epsilon_r_GT))), tf.float64), 2)

        print(iteration, total_field_err.numpy(), epsilon_err.numpy())

        total_field_re.append(total_field_err)
        epsilon_re.append(epsilon_err)

    """" 18. Process epsilon_r """

    epsilon_r_iter = epsilon_r_iter.numpy()

    # epsilon_r_iter[epsilon_r_iter < 1] = 1
    # epsilon_r_iter[epsilon_r_iter < 0j] = 0j

    """" 19. Plot epsilon_r """

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)

    original_real = ax1.imshow(np.real(epsilon_r_GT), cmap=plt.cm.jet, extent=[-doi_length/2, doi_length/2, -doi_length/2, doi_length/2])
    cb1 = fig.colorbar(original_real, ax=ax1, fraction=0.046, pad=0.04)
    cb1.ax.tick_params(labelsize=12)
    ax1.title.set_text(f"Original scatterer (real)")

    guess_real = ax2.imshow(np.real(epsilon_r_iter), cmap=plt.cm.jet, extent=[-doi_length/2, doi_length/2, -doi_length/2, doi_length/2])
    cb2 = fig.colorbar(guess_real, ax=ax2, fraction=0.046, pad=0.04)
    cb2.ax.tick_params(labelsize=12)
    ax2.title.set_text("Re(epsilon_r)")

    guess_imag = ax3.imshow(np.imag(epsilon_r_iter), cmap=plt.cm.jet, extent=[-doi_length/2, doi_length/2, -doi_length/2, doi_length/2])
    cb3 = fig.colorbar(guess_imag, ax=ax3, fraction=0.046, pad=0.04)
    cb3.ax.tick_params(labelsize=12)
    ax3.title.set_text("Im(epsilon_r)")

    # plt.setp(ax1.get_xticklabels(), fontsize=12, horizontalalignment="left")
    # plt.setp(ax2.get_xticklabels(), fontsize=12, horizontalalignment="left")
    # plt.setp(ax3.get_xticklabels(), fontsize=12, horizontalalignment="left")

    plt.setp(ax1.get_yticklabels(), fontsize=12)
    plt.setp(ax2.get_yticklabels(), fontsize=12)
    plt.setp(ax3.get_yticklabels(), fontsize=12)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)
    plt.show()

    """" 20. Plot convergence figures """

    fig, (ax1, ax2) = plt.subplots(ncols=2)

    ax1.plot(range(0, iterations), total_field_re)
    ax1.title.set_text("Total field relative error")

    ax2.plot(range(0, iterations), epsilon_re)
    ax2.title.set_text("epsilon_r relative error")

    plt.show()
