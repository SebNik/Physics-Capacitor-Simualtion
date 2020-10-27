# this file is opening saved classes
import pickle

if __name__ == "__main__":
    # setting a path
    path = ''
    # path open an old pickle file
    cap = pickle.load(open(path, "rb", -1))
    print(cap.plate_neg.plot_matrix_particles())
