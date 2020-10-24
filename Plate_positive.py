# this file is working with the positive plate
import uuid
import numpy as np
from Particle import Particle


class Plate_Positive:
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
        # print("Bounding box point: ", self._p1, self._p2, self._p3, self._p4)
        # setting the z plane
        self.z_plane = self._p1[2]
        # setting number of particles
        self._n = n
        # setting the matrix
        self.matrix = []
        self.matrix_pos = []
        # setting coordinates for the spacing of the particles
        if not random:
            x_ps = np.linspace(min(self._p1, self._p2)[0], max(self._p1, self._p2)[0], self._n)
            y_ps = np.linspace(min(self._p1, self._p2)[1], max(self._p1, self._p2)[1], self._n)
            # print("The positions for the spacing particles: ", x_ps, y_ps)
            # iterating through positions
            for x in x_ps:
                row = []
                data = []
                for y in y_ps:
                    # print("coordinates: ", x, y)
                    row.append([x, y])
                    data.append(Particle(x=x, y=y, z=self.z_plane, type_c='+'))
                # adding the data into the matrix
                self.matrix.append(data)
                self.matrix_pos.append(row)
            # setting the list in array
            self.matrix = np.array(self.matrix)
            self.matrix_pos = np.array(self.matrix_pos)

        # print(self.matrix, '\n\n', self.matrix_pos)

    def __repr__(self):
        # printing name of class
        return "The class handles all actions of a single Plate"

    def __str__(self):
        # printing th object out for information
        return "This is a Plate : {0}, with a positive charge. The bounding box coordinates are: p1: {1}, p2: {2}, p3: {3}, p4: {4}, on the z-plane: {5}".format(self._id, self._p1, self._p2, self._p3, self._p4, self.z_plane)


if __name__ == "__main__":
    # getting class information
    print(Plate_Positive)
    # setting instance of single plate
    plate_pos = Plate_Positive(n=3, p1=[0, 0, 0], p2=[1, 1, 0])
    # printing all information about it
    print(plate_pos)
    # getting values
