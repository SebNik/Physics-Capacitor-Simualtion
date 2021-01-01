# this file is opening saved classes
import pickle
import numpy as np
import matplotlib.pyplot as plt
from Plate_Capacitor import Plate_Capacitor

if __name__ == "__main__":
    # setting a path
    path = 'D:\\Python\\Programme\\Physics-Capacitor-Simualtion\\resources\\exports\\01_01_2021__15_06_19_Test_0_032_d_16_nbins_30'
    # path open an old pickle file
    cap = pickle.load(open(path+'\\class.pickle', "rb", -1))
    # cap.set_self_path(path=path)
    # cap = Plate_Capacitor(n_neg=16, n_pos=16, p1=[0.01, 0.01], p2=[0.02, 0.02], plane_z_pos=[0.001],
    #                       plane_z_neg=[0.004],
    #                       random=False, name='Test_16_normal_new_field_lines_density')
    # cap.sim(end_stop=2e-05, t=0.0000002)
    # cap.plot_field_lines_from_data()
    cap.plate_neg.plot_density(nbins=30)
    # cap.plot_field_lines_integral_calculation_flatten(num_field_lines=50, nbins=30, x_plane=[0.015])
    # x = np.linspace(0.0001, cap.plate_neg.x_length - 0.0001, 3) + cap.p1[0]
    # print(x)
    # ar = cap.plot_field_lines_integral_calculation(num_field_lines=64, delta_m=0.000004, nbins=30, x_plane=x,
    #                                                show=False, logs=True, room=False, fake_dist=True,
    #                                                path_fake_dist="C:\\Users\\lordv\\Documents\\Data_Density_Plot.xlsx")
