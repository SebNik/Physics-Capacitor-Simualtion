# this file is working with the positive plate
import uuid
import numpy as np
from scipy.stats import kde
from Particle import Particle
import matplotlib.pyplot as plt


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
        else:
            x_ps = np.random.uniform(low=min(self._p1, self._p2)[0], high=max(self._p1, self._p2)[0], size=(self._n,))
            y_ps = np.random.uniform(low=min(self._p1, self._p2)[1], high=max(self._p1, self._p2)[1], size=(self._n,))
        # print("The positions for the spacing particles: ", x_ps, y_ps)
        # iterating through positions
        for x in x_ps:
            row = []
            data = []
            for y in y_ps:
                # print("coordinates: ", x, y)
                self.matrix_pos.append([x, y])
                data.append(Particle(x=x, y=y, z=self.z_plane, type_c='+'))
            # adding the data into the matrix
            self.matrix.append(data)
        # setting the list in array
        self.matrix = np.array(self.matrix)
        self.matrix_pos = np.array(self.matrix_pos)
        # print(self.matrix, '\n\n', self.matrix_pos)

    def get_info_of_particles(self):
        # printing out all ethe information of the particles
        for p in self.matrix:
            print(p)

    def plot_matrix_particles(self):
        # plotting the particles
        plt.figure(figsize=(7, 7), dpi=80, facecolor='w', edgecolor='b')
        x, y = np.meshgrid(self.matrix_pos[:, 0], self.matrix_pos[:, 1])
        plt.scatter(x, y, c='r')
        plt.show()

    def plot_density(self):
        # plotting the density of the points
        plt.figure(figsize=(7, 7), dpi=80, facecolor='w', edgecolor='b')
        x, y = np.meshgrid(self.matrix_pos[:, 0], self.matrix_pos[:, 1])
        nbins = 50
        k = kde.gaussian_kde(self.matrix_pos.T)
        xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        # plot a density
        # pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.viridis)
        # plt.colorbar()
        plt.show()

    def __repr__(self):
        # printing name of class
        return "The class handles all actions of a single Plate"

    def __str__(self):
        # printing th object out for information
        return "This is a Plate : {0}, with a positive charge. The bounding box coordinates are: p1: {1}, p2: {2}, p3: {3}, p4: {4}, on the z-plane: {5}".format(
            self._id, self._p1, self._p2, self._p3, self._p4, self.z_plane)


if __name__ == "__main__":
    # getting class information
    # print(Plate_Positive)
    # setting instance of single plate
    plate_pos = Plate_Positive(n=20, p1=[0, 0, 0], p2=[1, 1, 0], random=False)
    # printing all information about it
    # print(plate_pos)
    # getting values
    # plate_pos.get_info_of_particles()
    # plotting out particles
    # plate_pos.plot_matrix_particles()
    # plotting the density of the points
    plate_pos.plot_density()
