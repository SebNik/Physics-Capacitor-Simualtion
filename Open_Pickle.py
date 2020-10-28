# this file is opening saved classes
import pickle
from Plate_Capacitor import Plate_Capacitor


if __name__ == "__main__":
    # setting a path
    path = 'D:\\Python\\Programme\\Physics-Capacitor-Simualtion\\resources\\exports\\27_10_2020__23_35_54\\class.pickle'
    # path open an old pickle file
    cap = pickle.load(open(path, "rb", -1))
    print(cap.plate_neg.plot_density())
    print(cap.cal_electric_field())
