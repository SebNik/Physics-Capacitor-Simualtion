# this file is opening saved classes
import pickle
import numpy as np
import matplotlib.pyplot as plt
from Plate_Capacitor import Plate_Capacitor

if __name__ == "__main__":
    # setting a path
    path = 'D:\\Python\\Programme\\Physics-Capacitor-Simualtion\\resources\\exports\\27_10_2020__23_35_54\\class.pickle'
    # path open an old pickle file
    cap = pickle.load(open(path, "rb", -1))
    resolution = 10
    e_field, length = cap.cal_electric_field(resolution=resolution)
    # np.savez_compressed('resources/exports/27_10_2020__23_35_54/arr.npz', e_field)
    # e_field_max = np.where(e_field[:, 3] == e_field[:, 3].max())[0][0]
    # print(e_field[e_field_max])
    # e_field = np.load('resources/exports/27_10_2020__23_35_54/arr.npz')['arr_0']
    # reading in z planes offsets
    z = []
    for t in e_field:
        if t[2] not in z:
            z.append(t[2])
    print(z)
    for off in z:
        # data 2d plot
        data_2d_plot = []
        for r in e_field:
            if r[2] == off:
                data_2d_plot.append(r)
        data_2d_plot = np.array(data_2d_plot)
        a = data_2d_plot[data_2d_plot[:, 2].argsort()]  # First sort doesn't need to be stable.
        a = a[a[:, 1].argsort(kind='mergesort')]
        a = a[a[:, 0].argsort(kind='mergesort')]

        fig = plt.figure(figsize=(7, 7), dpi=80, facecolor='w', edgecolor='b')
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(a[:,0], a[:,1])  # `plot_surface` expects `x` and `y` data to be 2D
        ax.plot_surface(X, Y, a[:,3])

        plt.show()

        # image plotting
        image = a[:, 3].reshape(int(len(a) / int(resolution)), int(resolution))
        fig, ax = plt.subplots()
        ax.imshow(image, **{'extent': [0.01, 0.02, 0.01, 0.02]})
        plt.title(str(off))
        # plt.show()
        plt.savefig(
            'D:\\Python\\Programme\\Physics-Capacitor-Simualtion\\resources\\exports\\27_10_2020__23_35_54\\E_Field\\' + 'E_Field_' + str(
                off) + '_Res_' + str(resolution) + '.png', dpi=100)
