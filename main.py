from config import *

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.special import jv as bessel1
from scipy.special import hankel1
from scipy.io import loadmat, savemat

from initialization_layer import InitializationLayer
from iteration_layer import DRIMLayer
from load_data import LoadData


if __name__ == '__main__':

    tf.random.set_seed(1234)
    tf.keras.backend.clear_session()
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    session = InteractiveSession(config=config)

    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

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

    """" 2. Adaptive regularization config parameters """

    iterations = 5
    reg_param = 0.01

    """" 3. Direct field from transmitter to receiver in free space """

    tx_xcoord = [pos[0] for pos in sensor_positions]
    tx_ycoord = [pos[1] for pos in sensor_positions]

    rx_xcoord = [pos[0] for pos in sensor_positions]
    rx_ycoord = [pos[1] for pos in sensor_positions]

    [xtd, xrd] = np.meshgrid(tx_xcoord, rx_xcoord)
    [ytd, yrd] = np.meshgrid(tx_ycoord, rx_ycoord)
    dist = np.sqrt((xtd - xrd) ** 2 + (ytd - yrd) ** 2)
    direct_field = (1j / 4) * hankel1(0, wave_number * dist)

    """" 3. Direct power from transmitter to receiver in free space """

    def remove_nan(field):
        np.fill_diagonal(field, np.nan)
        k = field.reshape(field.size, order='F')
        l = [x for x in k if not np.isnan(x)]
        m = np.transpose(np.reshape(l, (tx_count, rx_count - 1)))
        return m

    direct_field_nan = np.copy(direct_field)
    direct_field_nan = remove_nan(direct_field_nan)

    def get_power(field):
        power = ((np.abs(field) + noise_level) ** 2) * (wavelength ** 2) / (4 * np.pi * impedance)
        power = 10 * np.log10(power / 1e-3)
        return power

    # direct_power = get_power(direct_field_nan)
    direct_power = loadmat(r"C:\Users\dsamr\OneDrive - HKUST Connect\MPHIL RESEARCH\DATA\DRIM_ADAPTIVE_3\measurement_empty\direct_power.mat")["direct_power"]

    """" 4. Incident field from transmitter on all DoI grids """

    grid_xcoord = grid_positions[0]
    grid_xcoord = grid_xcoord.reshape(grid_xcoord.size, order='F')

    grid_ycoord = grid_positions[1]
    grid_ycoord = grid_ycoord.reshape(grid_ycoord.size, order='F')

    [xti, xsi] = np.meshgrid(tx_xcoord, grid_xcoord)
    [yti, ysi] = np.meshgrid(tx_ycoord, grid_ycoord)

    dist = np.sqrt((xti - xsi) ** 2 + (yti - ysi) ** 2)
    incident_field = (1j / 4) * hankel1(0, wave_number * dist)

    """" 5. Free space Green's function & scaled by a constant """

    [xts, xss] = np.meshgrid(tx_xcoord, grid_xcoord)
    [yts, yss] = np.meshgrid(tx_ycoord, grid_ycoord)

    dist = np.sqrt((xts - xss) ** 2 + (yts - yss) ** 2)

    G_freespace = (1j / 4) * hankel1(0, wave_number * dist)
    G_freespace_scaled = -impedance * np.pi * (grid_radius / 2) * \
                         bessel1(1, wave_number * grid_radius) * \
                         hankel1(0, wave_number * np.transpose(dist))


    def get_model():

        total_power = tf.keras.layers.Input((39, 40), dtype=tf.float32)

        epr_init, chi_init = InitializationLayer(reg_param,
                                                 m,
                                                 grid_positions,
                                                 direct_field,
                                                 direct_power,
                                                 incident_field,
                                                 dist)([total_power])

        epr_, chi_ = epr_init, chi_init
        total_power_ = total_power

        for i in range(0, iterations):
            epr_, chi_, = DRIMLayer(reg_param,
                                    m,
                                    grid_positions,
                                    direct_field,
                                    incident_field,
                                    G_freespace,
                                    G_freespace_scaled)([epr_, chi_, total_power_])
        model = tf.keras.Model(inputs=[total_power], outputs=[epr_, chi_])
        return model

    def load_data():

        batch_size = 1
        real_data_train, real_data_test, measurements_train, measurements_test = LoadData.main()

        train_dataset = tf.data.Dataset.from_tensor_slices((real_data_train, measurements_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((real_data_test, measurements_test))
        train_dataset = train_dataset.batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)

        return train_dataset, test_dataset

    model = get_model()
    # train_dataset, test_dataset = load_data()
    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)


    def train():

        steps = 1000
        steps = tf.cast(steps, tf.int64)

        train_ds = train_dataset.repeat(5).as_numpy_iterator()

        for step in tf.range(steps):

            step = tf.cast(step, tf.int64)

            gt_batch, total_power_batch = train_ds.next()

            with tf.GradientTape() as tape:
                out_batch, _, _ = model([total_power_batch])
                a = tf.abs(out_batch)
                a = tf.cast(a, tf.float32)

                b = tf.abs(gt_batch[0])
                b = tf.cast(b, tf.float32)

                loss = tf.reduce_mean(tf.square(a - b))
            network_gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(network_gradients, model.trainable_variables))

            if step % 10 == 0:
                tf.print("Step: ", step, "Loss: ", loss)

    # train()

    total_power = loadmat(r"C:\Users\dsamr\OneDrive - HKUST Connect\MPHIL RESEARCH\DATA\DRIM_ADAPTIVE_3\measurement_scatterer\0.mat")["total_power"]

    epr, chi = model([total_power])

    epsilon_r_GT = loadmat(r"C:\Users\dsamr\OneDrive - HKUST Connect\MPHIL RESEARCH\DATA\DRIM_ADAPTIVE_3\scatterer_data_inverse_20\0.mat")['scatterer']

    # """" Construct ground truth profile """
    #
    # size = 0.015
    # permittivity = 3.3
    # center_x = 0
    # center_y = 0
    #
    # # Circle
    # # epsilon_r_GT[(grid_positions[0] - -0.005) ** 2 + (grid_positions[1] - 0.045) ** 2 <= size ** 2] = permittivity
    # # epsilon_r_GT[(grid_positions[0] - -0.012) ** 2 + (grid_positions[1] + 0.045) ** 2 <= size ** 2] = permittivity
    #
    # mask = ((grid_positions[0] <= center_x + 0.04) & (grid_positions[0] >= center_x -0.04) &
    #         (grid_positions[1] <= center_y + 0.015) & (grid_positions[1] >= center_y - 0.015))
    # epsilon_r_GT[mask] = permittivity
    #
    # """" 19. Plot epsilon_r """
    #

    epsilon_r_iter = epr.numpy()

    # epsilon_r_iter[epsilon_r_iter < 1] = 1
    # epsilon_r_iter[epsilon_r_iter < 0j] = 0j

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)

    original_real = ax1.imshow(np.real(epsilon_r_GT), cmap=plt.cm.jet,
                               extent=[-doi_length / 2, doi_length / 2, -doi_length / 2, doi_length / 2])
    cb1 = fig.colorbar(original_real, ax=ax1, fraction=0.046, pad=0.04)
    cb1.ax.tick_params(labelsize=12)
    ax1.title.set_text(f"Original scatterer (real)")

    guess_real = ax2.imshow(np.real(epsilon_r_iter), cmap=plt.cm.jet,
                            extent=[-doi_length / 2, doi_length / 2, -doi_length / 2, doi_length / 2])
    cb2 = fig.colorbar(guess_real, ax=ax2, fraction=0.046, pad=0.04)
    cb2.ax.tick_params(labelsize=12)
    ax2.title.set_text("Re(epsilon_r)")

    guess_imag = ax3.imshow(np.imag(epsilon_r_iter), cmap=plt.cm.jet,
                            extent=[-doi_length / 2, doi_length / 2, -doi_length / 2, doi_length / 2])
    cb3 = fig.colorbar(guess_imag, ax=ax3, fraction=0.046, pad=0.04)
    cb3.ax.tick_params(labelsize=12)
    ax3.title.set_text("Im(epsilon_r)")

    plt.setp(ax1.get_yticklabels(), fontsize=12)
    plt.setp(ax2.get_yticklabels(), fontsize=12)
    plt.setp(ax3.get_yticklabels(), fontsize=12)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)
    plt.show()


    ##

    layers = [layer.output for layer in model.layers[1:]]
    layers_model = tf.keras.Model(inputs=model.inputs, outputs=layers)
    out = layers_model([total_power])
    epr_init = out[0][0].numpy()
    epr = [epr_init]
    for item in out[1:]:
        epr.append(item[0].numpy())



