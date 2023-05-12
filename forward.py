from config import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv as bessel1
from scipy.special import hankel1
from scipy.io import savemat

if __name__ == '__main__':

    """" Forward problem config parameters """

    m = 120
    grid_length = doi_length / m
    num_grids = m ** 2

    # """ Grid positions/ centroids """
    centroids_x = np.arange(start=- doi_length / 2 + grid_length / 2, stop=doi_length / 2, step=grid_length)
    centroids_y = np.arange(start=doi_length / 2 - grid_length / 2, stop=-doi_length / 2, step=-grid_length)
    grid_positions = np.meshgrid(centroids_x, centroids_y)

    # """ Grid radius """
    grid_radius = np.sqrt(grid_length ** 2 / np.pi)

    # """ Constants to be used in code """
    C1 = -impedance * np.pi * (grid_radius / 2)
    C2 = bessel1(1, wave_number * grid_radius)
    C3 = hankel1(1, wave_number * grid_radius)

    """" 1. DoI permittivity profile """
    size = 0.015
    permittivity = 3.3
    center_x = 0
    center_y = 0

    scatterer = np.ones((m, m), dtype=complex)

    # Circle
    # scatterer[(grid_positions[0] - -0.005) ** 2 + (grid_positions[1] - 0.045) ** 2 <= size** 2] = permittivity
    # scatterer[(grid_positions[0] - -0.012) ** 2 + (grid_positions[1] + 0.045) ** 2 <= size** 2] = permittivity

    # scatterer[(grid_positions[0] - -0.005) ** 2 + (grid_positions[1] - 0.045) ** 2 <= size** 2] = 1
    # scatterer[(grid_positions[0] - -0.012) ** 2 + (grid_positions[1] + 0.045) ** 2 <= size** 2] = 1

    # # Rectangle
    mask = ((grid_positions[0] <= center_x + 0.04) & (grid_positions[0] >= center_x -0.04) &
            (grid_positions[1] <= center_y + 0.015) & (grid_positions[1] >= center_y - 0.015))
    scatterer[mask] = permittivity

    # Plot
    plt.figure(1)
    plt.imshow(np.real(scatterer), cmap=plt.cm.jet, extent=[-doi_length/2, doi_length/2, -doi_length/2, doi_length/2])
    plt.colorbar()
    plt.show()

    """" 2. Direct field from transmitter to receiver in free space """

    tx_xcoord = [pos[0] for pos in sensor_positions]
    tx_ycoord = [pos[1] for pos in sensor_positions]

    rx_xcoord = [pos[0] for pos in sensor_positions]
    rx_ycoord = [pos[1] for pos in sensor_positions]

    [xtd, xrd] = np.meshgrid(tx_xcoord, rx_xcoord)
    [ytd, yrd] = np.meshgrid(tx_ycoord, rx_ycoord)
    dist = np.sqrt((xtd - xrd) ** 2 + (ytd - yrd) ** 2)
    direct_field = (1j / 4) * hankel1(0, wave_number * dist)

    """" 3. Incident field from transmitter on all DoI grids """

    grid_xcoord = grid_positions[0]
    grid_xcoord = grid_xcoord.reshape(grid_xcoord.size, order='F')

    grid_ycoord = grid_positions[1]
    grid_ycoord = grid_ycoord.reshape(grid_ycoord.size, order='F')

    [xti, xsi] = np.meshgrid(tx_xcoord, grid_xcoord)
    [yti, ysi] = np.meshgrid(tx_ycoord, grid_ycoord)

    dist = np.sqrt((xti - xsi) ** 2 + (yti - ysi) ** 2)
    incident_field = (1j / 4) * hankel1(0, wave_number * dist)

    """" 4. Grids containing object """

    unrolled_scatterer = scatterer.reshape(scatterer.size, order='F')
    object_grids = np.nonzero(unrolled_scatterer != 1)
    object_grids = object_grids[0]

    """" 5. Method of Moment """

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

    """" 6. Induced current on every point in the DoI """

    field_on_object = -incident_field[object_grids]
    J1 = np.linalg.inv(Z) @ field_on_object

    induced_current = np.zeros((m ** 2, tx_count), dtype=complex)
    for i in range(len(object_grids)):
        induced_current[object_grids[i], :] = J1[i, :]

    """" 7. Free space Green's function scaled by a constant """

    [xts, xss] = np.meshgrid(tx_xcoord, grid_xcoord)
    [yts, yss] = np.meshgrid(tx_ycoord, grid_ycoord)

    dist = np.sqrt((xts - xss)**2 + (yts - yss)**2)

    G_scaled = - impedance * np.pi * (grid_radius / 2) * bessel1(1, wave_number * grid_radius) * hankel1(0, wave_number * np.transpose(dist))

    """" 7. Scattered field collected at the receivers """

    scattered_field = G_scaled @ induced_current

    """" 8. Total field at receivers """

    def remove_nan(field):
        np.fill_diagonal(field, np.nan)
        k = field.reshape(field.size, order='F')
        l = [x for x in k if not np.isnan(x)]
        m = np.transpose(np.reshape(l, (tx_count, rx_count - 1)))
        return m

    total_field = direct_field + scattered_field

    direct_field = remove_nan(direct_field)
    total_field = remove_nan(total_field)

    """" 9. RSS values at receivers """

    def get_power(field):
        power = ((np.abs(field) + noise_level) ** 2) * (wavelength ** 2) / (4 * np.pi * impedance)
        power = 10 * np.log10(power / 1e-3)
        return power

    direct_power = get_power(direct_field)
    total_power = get_power(total_field)

    """" 10. Save data """

    savemat('data/scatterer.mat', {"scatterer": scatterer})
    savemat('data/direct_field.mat', {"direct_field": direct_field})
    savemat('data/direct_power.mat', {"direct_power": direct_power})
    savemat('data/scattered_field.mat', {"scattered_field": scattered_field})
    savemat('data/total_field.mat', {"total_field": total_field})
    savemat('data/total_power.mat', {"total_power": total_power})
