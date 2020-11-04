# this file is opening saved classes
import pickle
import numpy as np
import matplotlib.pyplot as plt
from Plate_Capacitor import Plate_Capacitor

if __name__ == "__main__":
    # setting a path
    path = 'D:\\Python\\Programme\\Physics-Capacitor-Simualtion\\resources\\exports\\29_10_2020__12_15_19\\class.pickle'
    # path open an old pickle file
    cap = pickle.load(open(path, "rb", -1))
    resolution = 80
    z = list(np.linspace(0.001, 0.002, 20))[1:]
    cap.analysis_2D(resolution=resolution, z_plane=z, size=1.75)
    # e_field, length, forces = cap.cal_electric_field(resolution=resolution)
    # e_field = np.load('resources/exports/29_10_2020__12_15_19/e_field_array.npz', allow_pickle=True)['arr_0']
    # e_forces = np.load('resources/exports/29_10_2020__12_15_19/forces_array.npz', allow_pickle=True)['arr_0']
    #
    # forces = e_forces[:, 3].reshape(int(len(e_forces) / int(resolution)), int(resolution))
    # fig, ax = plt.subplots()
    # m = ax.imshow(forces)
    # fig.colorbar(m)
    # plt.show()

    # image getting the 2d
    # image = e_field[:, 3].reshape(int(len(e_field) / int(resolution)), int(resolution))
    # print(image)
    # for i in range(0, resolution, 10):
    #     plt.title(str(i))
    #     plt.plot(image[i])
    #     plt.show()

    # for i in range(0, resolution, 25):
    #     plt.plot(image[i], label=str(i))
    # plt.title('Full')
    # plt.legend(loc='upper right')
    # plt.show()

    # getting max and min for plots
    # max_v = max(e_field[:, 3])
    # min_v = min(e_field[:, 3])
    # # image plotting in 2d
    # fig, ax = plt.subplots()
    # m = ax.imshow(image, vmin=min_v, vmax=max_v)
    # fig.colorbar(m)
    # plt.show()
