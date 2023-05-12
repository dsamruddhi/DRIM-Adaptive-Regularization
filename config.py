import numpy as np

""" System parameters """
frequency = 2e9
wavelength = 3e8 / frequency
wave_number = 2 * np.pi / wavelength
impedance = 120 * np.pi

""" DoI parameters """
doi_length = wavelength

""" Sensor parameters """
geometry = "circle"
rx_count = 40
tx_count = 40

""" Sensor positions """
x = 5 * wavelength
y = x
angles = np.arange(start=0, stop=360, step=360/rx_count)
theta = [np.deg2rad(angle) for angle in angles]
sensor_x = x*np.cos(theta)
sensor_y = y*np.sin(theta)
sensor_positions = np.transpose(np.array([sensor_x, sensor_y]))

""" Sensor links """
sensor_links = []
for i in range(rx_count):
    for j in range(rx_count):
        if i != j:
            sensor_links.append((i, j))
num_links = len(sensor_links)

""" Other parameters """
nan_remove = True
noise_level = 0
