# this file is working with the negative plate
import uuid
import numpy as np
from scipy.stats import kde
from Particle import Particle
import matplotlib.pyplot as plt


class Plate_Negative:
    # this is a class which represents one negative plate
    # this plate is filled with electrons which can move freely
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

    def get_inner_forces(self, n=False):
        if not n:
            n = len(self.matrix.flatten())
        # getting the inner force of the plate
        forces_list = []
        forces_dic = {}
        # iterating through electrons
        for e_cal in self.matrix.flatten():
            force_sum = np.array([0.0, 0.0, 0.0])
            count = 0
            for e_check in self.matrix.flatten():
                if e_cal != e_check:
                    force, force_vector, force_vector_x, force_vector_y, force_vector_z = e_cal.cal_force(
                        particle=e_check)
                    # print("Forces: from: ", str(e_cal.get_id()), 'to: ', str(e_check.get_id()), '--->', force,
                    #       force_vector, force_vector_x, force_vector_y, force_vector_z)
                    force_sum += force_vector
                    count += 1
                    if count >= n:
                        break
            # print("Force sum: from: ", str(e_cal.get_id()), ' to: ', str(e_check.get_id()), ' --->', force_sum)
            forces_list.append(force_sum)
            forces_dic[str(e_cal.get_id())] = force_sum
        # returning values
        return forces_list, forces_dic

    def plot_matrix_particles(self):
        # plotting the particles
        plt.figure(figsize=(7, 7), dpi=80, facecolor='w', edgecolor='b')
        x, y = np.meshgrid(self.matrix_pos[:, 0], self.matrix_pos[:, 1])
        plt.scatter(x, y, c='r')
        plt.show()

    def plot_matrix_particles_vector(self, n=False):
        # plotting the particles and inner force vectors
        # setting figure
        plt.figure(figsize=(7, 7), dpi=80, facecolor='w', edgecolor='b')
        # getting forces data
        f_list, f_dic = self.get_inner_forces(n=n)
        print(f_dic)
        for e in self.matrix.flatten():
            plt.quiver(e.get_x(), e.get_y(), f_dic[str(e.get_id())][0], f_dic[str(e.get_id())][1], hatch='o',
                       width=0.01)
        # getting x,y for particles plot
        x, y = np.meshgrid(self.matrix_pos[:, 0], self.matrix_pos[:, 1])
        # plotting particles
        plt.scatter(x, y, c='r')
        # showing the plot
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
        return "This is a Plate : {0}, with a negative charge. The bounding box coordinates are: p1: {1}, p2: {2}, p3: {3}, p4: {4}, on the z-plane: {5}".format(
            self._id, self._p1, self._p2, self._p3, self._p4, self.z_plane)


if __name__ == "__main__":
    # getting class information
    print(Plate_Negative)
    # setting instance of single plate
    plate_neg = Plate_Negative(n=5, p1=[0, 0, 0], p2=[0.001, 0.001, 0], random=True)
    # printing all information about it
    # print(plate_neg)
    # getting values
    # plate_neg.get_info_of_particles()
    # plotting out particles
    # plate_neg.plot_matrix_particles()
    # plotting the density of the points
    # plate_neg.plot_density()
    # getting the inner forces
    # print(plate_neg.get_inner_forces())
    # plotting inner forces
    plate_neg.plot_matrix_particles_vector()
