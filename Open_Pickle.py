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
    cap = Plate_Capacitor(n_neg=7, n_pos=7, p1=[0.01, 0.01], p2=[0.02, 0.02], plane_z_pos=[0.001],
                          plane_z_neg=[0.002],
                          random=False)
    cap.plotting_plates()
    delta = cap.plate_neg.x_length * (1.5 - 1) / 2
    x = np.linspace(0 - delta, cap.plate_neg.x_length + delta, 100) + cap._p1[0]
    # starting analysis for the 3d room
    # res_2d = 100
    # res_3d = 70
    # cap.analysis(resolution_2d=res_2d, resolution_3d=res_3d, size=1.5)
    # getting the field lines
    print(cap.plot_field_lines(x_plane=x[int(len(x) / 8)]))
