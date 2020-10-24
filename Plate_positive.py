# this file is working with the positive plate
import uuid
import numpy as np


class Plate_positive:
    # this is a class which represents one positive plate
    # here are all the protons fixed anc can't move
    def __init__(self, n, p1, p2, random=False):
        # setting an id for the plate
        self._id = uuid.uuid4()
        # setting bounding box edges
        self._p1 = p1
        self._p2 = p2
        self._p3 = [p1[0], p2[1], p1[2]]
        self._p4 = [p2[0], p1[1], p1[2]]
        print("Bounding box point: ", self._p1, self._p2, self._p3, self._p4)
        # setting number of particles
        self._n = n

    def __repr__(self):
        # printing name of class
        return "The class handles all actions of a single Plate"

    def __str__(self):
        # printing th object out for information
        return "This is a Plate : {0}, with a positive charge. The bounding box coordinates are: p1: {1}, p2: {2}, p3: {3}, p4: {4}".format(
            self._id, self._p1, self._p2, self._p3, self._p4)


if __name__ == "__main__":
    # getting class information
    print(Plate_positive)
    # setting instance of single plate
    plate_pos = Plate_positive(n=3, p1=[0, 0, 0], p2=[3, 3, 0])
    # printing all information about it
    print(plate_pos)
    # getting values
