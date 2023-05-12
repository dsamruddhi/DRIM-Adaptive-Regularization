from config import *

import numpy as np
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
    iterations = 5
    reg_type = "ridge"
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

    epsilon_r_GT = loadmat(r"C:\Users\dsamr\OneDrive - HKUST Connect\MPHIL RESEARCH\DATA\DRIM_ADAPTIVE_3\scatterer_data_inverse_20\0.mat")['scatterer']

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

    """" 8. Initialization: Get Rytov data """

    data_init = (total_power - direct_power) / (20 * np.log10(np.exp(1)))
    data_init = data_init.reshape(data_init.size, order='F')

    """" 9. Initialization: Ridge regression """

    dim = H_init.shape[1]
    lambda_max = np.linalg.norm(np.transpose(H_init) @ data_init, 2)
    chi_init = np.linalg.inv((H_init.T @ H_init) + lambda_max * reg_param * np.eye(dim)) @ H_init.T @ data_init

    """" 16. Initialization: Reshape chi """

    chi_r = chi_init[:m ** 2]
    chi_r = np.reshape(chi_r, (m, m), order='F')

    chi_i = chi_init[m ** 2:]
    chi_i = np.reshape(chi_i, (m, m), order='F')

    chi_init = chi_r + 1j * chi_i
    epsilon_r_init = chi_init + 1

    """" 7. Loop - { [forward, inverse] - [forward, inverse] ... } """

    epsilon_r_iter = epsilon_r_init
    chi = chi_init

    for iteration in range(0, iterations):

        # forward code

        """" 8. Forward: Grids containing object """

        unrolled_scatterer = epsilon_r_iter.reshape(epsilon_r_iter.size, order='F')
        object_grids = np.nonzero(unrolled_scatterer != 1)
        object_grids = object_grids[0]

        """" 9. Forward: Method of Moment """

        Z = np.zeros((len(object_grids), len(object_grids)), dtype=np.complex64)
        unroll_x = grid_positions[0].reshape(grid_positions[0].size, order='F')
        unroll_y = grid_positions[1].reshape(grid_positions[1].size, order='F')
        obj_x = unroll_x[object_grids]
        obj_y = unroll_y[object_grids]

        for index, value in enumerate(object_grids):
            x_incident = obj_x[index]
            y_incident = obj_y[index]

            dipole_distances = np.sqrt((x_incident - obj_x) ** 2 + (y_incident - obj_y) ** 2)

            a1 = hankel1(0, wave_number * dipole_distances)
            b1 = impedance * unrolled_scatterer[value] / (wave_number * (unrolled_scatterer[value] - 1))

            z1 = C1 * C2 * a1
            z1[index] = C1 * C3 - 1j * b1

            assert len(z1) == len(dipole_distances)
            Z[index, :] = z1

        """" 10. Forward: Induced current due to object with epsilon_r, non-freespace incident field, 
        inhomogeneous Green's function """

        field_on_object = -incident_field[object_grids]
        G_object_to_rx = -G_freespace[object_grids]

        # Initialize
        incident_field_iter = np.copy(incident_field)
        G_iter = np.copy(G_freespace)
        induced_current = np.zeros((m ** 2, tx_count), dtype=complex)

        # induced current due to tx for estimating total field [tx DoI interaction]
        J1 = np.linalg.inv(Z) @ field_on_object
        # induced current due to rx as pseudo source for estimating inhomogenous Green's function [DoI rx interaction]
        G_J1 = np.linalg.inv(Z) @ G_object_to_rx

        for i in range(len(object_grids)):
            induced_current[object_grids[i], :] = J1[i, :]

            # Quantities used during inverse
            incident_field_iter[object_grids[i], :] = (1j * impedance * J1[object_grids[i], :]) / ((unrolled_scatterer[object_grids[i]] - 1) * wave_number)
            G_iter[object_grids[i], :] = (1j * impedance * G_J1[object_grids[i], :]) / ((unrolled_scatterer[object_grids[i]] - 1) * wave_number)

        G_iter = np.transpose(G_iter)

        """" 11. Forward: Scattered field collected at the receivers hen object with epsilon_r is kept in the DoI"""

        scattered_field_iter = G_freespace_scaled @ induced_current

        """" 12. Forward: Total field at receiver when object with epsilon_r is kept in the DoI"""

        def remove_nan(field):
            np.fill_diagonal(field, np.nan)
            k = field.reshape(field.size, order='F')
            l = [x for x in k if not np.isnan(x)]
            m = np.transpose(np.reshape(l, (tx_count, rx_count - 1)))
            return m

        total_field_iter = direct_field + scattered_field_iter

        # inverse code

        """" 13. Inverse: Rytov model """

        A = np.zeros((len(sensor_links), num_grids), dtype=complex)

        for i, pair in enumerate(sensor_links):
            A[i, :] = grid_area * (wave_number**2) * np.divide(np.multiply(G_iter[pair[1], :], np.transpose(incident_field_iter[:, pair[0]])),
                                                            total_field_iter[pair[1], pair[0]])
        A_real = np.real(A)
        A_imag = np.imag(A)

        H_iter = np.concatenate((A_real, -A_imag), axis=1)

        """" 14. Forward/Inverse: RSS values at receiver """

        total_field_iter = remove_nan(total_field_iter)

        def get_power(field):
            power = ((np.abs(field) + noise_level) ** 2) * (wavelength ** 2) / (4 * np.pi * impedance)
            power = 10 * np.log10(power / 1e-3)
            return power

        total_power_iter = get_power(total_field_iter)

        """" 14. Inverse: Rytov data """

        data_iter = (total_power - total_power_iter) / (20 * np.log10(np.exp(1)))
        data_iter = data_iter.reshape(data_iter.size, order='F')

        """" 15. Inverse: Ridge regression """

        dim = H_iter.shape[1]
        lambda_max = np.linalg.norm(np.transpose(H_iter) @ data_iter, 2)
        chi_iter = np.linalg.inv((H_iter.T @ H_iter) + lambda_max * reg_param * np.eye(dim)) @ H_iter.T @ data_iter

        """" 16. Inverse: Reshape chi """

        delta_chi_r = chi_iter[:m ** 2]
        delta_chi_r = np.reshape(delta_chi_r, (m, m), order='F')

        delta_chi_i = chi_iter[m ** 2:]
        delta_chi_i = np.reshape(delta_chi_i, (m, m), order='F')

        delta_chi = delta_chi_r + 1j * delta_chi_i
        chi = chi + delta_chi
        epsilon_r_iter = chi + 1

        """" 17. Evaluation criteria - relative errors for total field and epsilon_r """

        epr = np.copy(epsilon_r_iter)

        # Relative error for total field
        total_field_err = np.linalg.norm(np.abs(total_field.reshape(total_field.size))
                                         - np.abs(total_field_iter.reshape(total_field_iter.size)), 1) \
                          / np.linalg.norm(np.abs(total_field.reshape(total_field.size)), 1)

        # Relative error for epsilon_r
        epsilon_err = np.linalg.norm(np.abs(epsilon_r_GT.reshape(epsilon_r_GT.size))
                                     - np.abs(epr.reshape(epr.size)), 2) \
                      / np.linalg.norm(np.abs(epsilon_r_GT.reshape(epsilon_r_GT.size)), 2)

        print(iteration, total_field_err, epsilon_err)

        total_field_re.append(total_field_err)
        epsilon_re.append(epsilon_err)

    """" 18. Process epsilon_r """

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
