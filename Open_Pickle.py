# this file is opening saved classes
import pickle
import numpy as np
import matplotlib.pyplot as plt
from Plate_Capacitor import Plate_Capacitor

if __name__ == "__main__":
    # setting a path
    path = 'D:\\Python\\Programme\\Physics-Capacitor-Simualtion\\resources\\exports\\21_11_2020__15_17_43_integral\\class.pickle'
    # path open an old pickle file
    cap = pickle.load(open(path, "rb", -1))
    # cap = Plate_Capacitor(n_neg=10, n_pos=10, p1=[0.01, 0.01], p2=[0.02, 0.02], plane_z_pos=[0.001],
    #                       plane_z_neg=[0.004],
    #                       random=False)
    # cap.sim()
    # cap.plate_neg.plot_density_distribution(nbins=300)
    # cap.plate_neg.plot_density()
    x = np.linspace(0.0001, cap.plate_neg.x_length - 0.0001, 3) + cap.p1[0]
    cap.plot_field_lines_integral_calculation(x_plane=x)

    # getting the field lines
    # cap.set_self_path(path='D:\\Python\\Programme\\Physics-Capacitor-Simualtion\\resources\\exports\\21_11_2020__15_17_43_bigger_d')
    # cap.plate_neg.move_plane_on_z_plane(new_z_plane=0.015)
    # cap.plot_field_lines_static(num_field_lines=15)
    # x = np.linspace(0.0001, cap.plate_neg.x_length - 0.0001, 3) + cap.p1[0]
    # ar = cap.plot_field_lines(x_plane=x, num_field_lines=15, delta_m=0.00001)
