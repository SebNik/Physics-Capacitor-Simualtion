# this file is opening saved classes
import pickle
import numpy as np
import matplotlib.pyplot as plt
from Plate_Capacitor import Plate_Capacitor

if __name__ == "__main__":
    # setting a path
    path = 'D:\\Python\\Programme\\Physics-Capacitor-Simualtion\\resources\\exports\\29_10_2020__12_15_19\\class.pickle'
    # path open an old pickle file
    # cap = pickle.load(open(path, "rb", -1))
    resolution = 100
    # cap.analysis_2D(resolution=resolution, z_plane=0.00195)
    # e_field, length, forces = cap.cal_electric_field(resolution=resolution)
    e_field = np.load('resources/exports/29_10_2020__12_15_19/e_field_array.npz')['arr_0']

    plt.plot(e_field[:, 3])
    plt.show()

    # getting max and min for plots
    max_v = max(e_field[:, 3])
    min_v = min(e_field[:, 3])
    # image getting the 2d
    image = e_field[:, 3].reshape(int(len(e_field) / int(resolution)), int(resolution))
    print(image)
    # image plotting in 2d
    fig, ax = plt.subplots()
    m = ax.imshow(image, vmin=min_v, vmax=max_v)
    fig.colorbar(m)
    plt.show()
