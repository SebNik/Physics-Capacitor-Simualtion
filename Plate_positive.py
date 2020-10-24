# this file is working with the positive plate
import numpy as np


class Plate_positive:
    # this is a class which represents one positive plate
    # here are all the protons fixed anc can't move
    def __init__(self, n, p1, p2, random=False):
        # setting bounding box edges
        p3 = [p1[0], p2[1], p1[2]]
        p4 = [p2[0], p1[1], p1[2]]
        print(p1, p2, p3, p4)
        # setting number of particles
        self._n = n


if __name__ == "__main__":
    # getting class information
    print(Plate_positive)
    # setting instance of single plate
    plate_pos = Plate_positive(n=3, p1=[0, 0, 0], p2=[3, 3, 0])
    # printing all information about it
    print(plate_pos)
    # getting values
