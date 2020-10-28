# this file is opening saved classes
import pickle
import numpy as np
from Plate_Capacitor import Plate_Capacitor

if __name__ == "__main__":
    # setting a path
    path = 'D:\\Python\\Programme\\Physics-Capacitor-Simualtion\\resources\\exports\\27_10_2020__23_35_54\\class.pickle'
    # path open an old pickle file
    cap = pickle.load(open(path, "rb", -1))
    # print(cap.plate_neg.plot_density())
    e_field, length = cap.cal_electric_field()
    np.savez_compressed('resources/exports/27_10_2020__23_35_54/arr.npz', e_field)
    e_field_max = np.where(e_field[:, 3] == e_field[:, 3].max())[0][0]
    # print(e_field[e_field_max])
