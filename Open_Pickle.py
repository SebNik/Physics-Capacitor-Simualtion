# this file is opening saved classes
import pickle
import numpy as np
import matplotlib.pyplot as plt
from Plate_Capacitor import Plate_Capacitor

if __name__ == "__main__":
    # setting a path
    # path = 'D:\\Python\\Programme\\Physics-Capacitor-Simualtion\\resources\\exports\\21_11_2020__15_17_43_fake_dist\\class.pickle'
    # # path open an old pickle file
    # cap = pickle.load(open(path, "rb", -1))
    cap = Plate_Capacitor(n_neg=10, n_pos=10, p1=[0.01, 0.01], p2=[0.02, 0.02], plane_z_pos=[0.001],
                          plane_z_neg=[0.004],
                          random=False)
    cap.sim()

    # cap.set_self_path(
    #     path='D:\\Python\\Programme\\Physics-Capacitor-Simualtion\\resources\\exports\\21_11_2020__15_17_43_fake_dist')
    # x = np.linspace(0.0001, cap.plate_neg.x_length - 0.0001, 3) + cap.p1[0]
    # print(x)
    # ar = cap.plot_field_lines_integral_calculation(num_field_lines=64, delta_m=0.000004, nbins=30, x_plane=x,
    #                                                show=False, logs=True, room=False, fake_dist=True,
    #                                                path_fake_dist="C:\\Users\\lordv\\Documents\\Data_Density_Plot.xlsx")
