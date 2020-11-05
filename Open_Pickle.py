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
    # starting analysis for the 3d room
    # res_2d = 100
    # res_3d = 70
    # cap.analysis(resolution_2d=res_2d, resolution_3d=res_3d, size=1.5)
    # getting the field lines
    cap.plot_field_lines()
