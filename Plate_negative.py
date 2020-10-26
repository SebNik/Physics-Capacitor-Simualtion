# this file is working with the negative plate
import uuid
import numpy as np
from scipy.stats import kde
from Particle import Particle
import matplotlib.pyplot as plt
from scipy.constants import electron_mass


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
        # getting length for plate
        self._x_length = abs(p2[0] - p1[0])
        self._y_length = abs(p2[1] - p1[1])
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
            # x_ps = np.random.uniform(low=min(self._p1, self._p2)[0], high=max(self._p1, self._p2)[0], size=(self._n,))
            # y_ps = np.random.uniform(low=min(self._p1, self._p2)[1], high=max(self._p1, self._p2)[1], size=(self._n,))
            x_ps = [np.random.random_sample() * self._x_length + min(self._p1, self._p2)[0] for i in range(n)]
            y_ps = [np.random.random_sample() * self._y_length + min(self._p1, self._p2)[1] for i in range(n)]
        # print("The positions for the spacing particles: ", x_ps, y_ps)
        # iterating through positions
        for x in x_ps:
            row = []
            data = []
            for y in y_ps:
                # print("coordinates: ", x, y)
                data.append(Particle(x=x, y=y, z=self.z_plane, type_c='-'))
                self.matrix_pos.append([data[-1].get_x(), data[-1].get_y()])
            # adding the data into the matrix
            self.matrix.append(data)
        # setting the list in array
        self.matrix = np.array(self.matrix)
        # self.matrix_pos = np.array(self.matrix_pos)
        # print(self.matrix, '\n\n', self.matrix_pos)

    def update_matrix_pos(self):
        # this function is updating the matrix_pos
        # setting it newly
        self.matrix_pos = []
        for e in self.matrix.flatten():
            self.matrix_pos.append([e.get_x(), e.get_y()])
        # setting it to numpy array
        # self.matrix_pos = np.array(self.matrix_pos)

    def get_info_of_particles(self):
        # printing out all ethe information of the particles
        for p in self.matrix:
            print(p)

    def get_inner_forces(self):
        # getting the inner force of the plate
        forces_list = []
        forces_dic = {}
        # iterating through electrons
        for e_cal in self.matrix.flatten():
            force_sum = np.array([0.0, 0.0, 0.0])
            for e_check in self.matrix.flatten():
                if e_cal != e_check:
                    # print(e_cal.get_id(), e_check.get_id(),e_cal.get_x(), e_check.get_x(), e_cal.get_y(),
                    # e_check.get_y(), e_cal.get_z(), e_check.get_z())
                    force, force_vector, force_vector_x, force_vector_y, force_vector_z = e_cal.cal_force(
                        particle=e_check)
                    # print("Forces: from: ", str(e_cal.get_id()), 'to: ', str(e_check.get_id()), '--->', force,
                    #       force_vector, force_vector_x, force_vector_y, force_vector_z)
                    force_sum += force_vector
                else:
                    # print(str(e_cal.get_id()), str(e_check.get_id()))
                    None
            # print("Force sum: from: ", str(e_cal.get_id()), ' to: ', str(e_check.get_id()), ' --->', force_sum)
            forces_list.append(force_sum)
            forces_dic[str(e_cal.get_id())] = force_sum
        # returning values
        return forces_list, forces_dic

    def move_by_force_vector(self, id, force, p=1):
        # this function is moving the particle with the id by the force vector in p
        for e in self.matrix.flatten():
            # found the right particle
            if str(e.get_id()) == id:
                # setting old position
                x_old = e.get_x()
                y_old = e.get_y()
                # setting new force vector
                new_force_vector = force * p
                # setting new position
                x_new = x_old + new_force_vector[0]
                y_new = y_old + new_force_vector[1]
                # setting state
                s = 1
                # checking if bigger than boundaries
                if x_new > self._p2[0]:
                    x_new = self._p2[0]
                    s = 0
                elif x_new < self._p1[0]:
                    x_new = self._p1[0]
                    s = 0
                if y_new > self._p2[0]:
                    y_new = self._p2[0]
                    s = 0
                elif y_new < self._p1[0]:
                    y_new = self._p1[0]
                    s = 0
                # moving the particle
                e.set_x(x=x_new)
                e.set_y(y=y_new)
                # print(x_old, y_old, e.get_x(), e.get_y())
        return s

    def move_by_force_time(self, id, force, delta_t=0.001):
        # this function is moving the particle with the id by the force vector in the time t
        # setting vars for case
        s, x_rel, y_rel, rel_avg = 0, 0, 0, 0
        for e in self.matrix.flatten():
            # found the right particle
            if str(e.get_id()) == id:
                # setting old position
                x_old = e.get_x()
                y_old = e.get_y()
                # setting new unit force vector
                unit_force = force / np.linalg.norm(force)
                # setting the abs distance
                d_abs = (force[0] ** 2 + force[1] ** 2) ** 0.5
                # finding out the s and the acceleration
                a = d_abs / electron_mass
                s = 0.5 * a * (delta_t ** 2)
                # setting the new force vector
                new_force_vector = unit_force * s
                # setting new position
                x_new = x_old + new_force_vector[0]
                y_new = y_old + new_force_vector[1]
                # setting state
                s = 1
                # checking if bigger than boundaries
                if x_new > self._p2[0]:
                    x_new = self._p2[0]
                    s = 0
                elif x_new < self._p1[0]:
                    x_new = self._p1[0]
                    s = 0
                if y_new > self._p2[0]:
                    y_new = self._p2[0]
                    s = 0
                elif y_new < self._p1[0]:
                    y_new = self._p1[0]
                    s = 0
                # updating the matrix
                self.update_matrix_pos()
                # checking if there is not another particle already there
                if [x_new, y_new] not in self.matrix_pos:
                    # moving the particle
                    e.set_x(x=x_new)
                    e.set_y(y=y_new)
                else:
                    s = 0
                # getting the relative change
                x_rel = x_new * 100 / x_old
                y_rel = y_new * 100 / y_old
                rel_avg = (x_rel + y_rel) / 2
                # print(x_old, y_old, e.get_x(), e.get_y())
        return s, x_rel, y_rel, rel_avg - 100

    def plot_matrix_particles(self, save=False, path=None, show=True):
        # plotting the particles
        plt.figure(figsize=(7, 7), dpi=80, facecolor='w', edgecolor='b')
        x = [e.get_x() for e in self.matrix.flatten()]
        y = [e.get_y() for e in self.matrix.flatten()]
        plt.scatter(x, y, c='r', alpha=0.1)
        if save:
            plt.savefig(path, dpi=100)
        if show:
            plt.show()
        plt.close()
        plt.clf()

    def plot_matrix_particles_vector(self):
        # plotting the particles and inner force vectors
        # setting figure
        plt.figure(figsize=(7, 7), dpi=80, facecolor='w', edgecolor='b')
        # getting forces data
        f_list, f_dic = self.get_inner_forces()
        print(f_dic)
        for e in self.matrix.flatten():
            plt.quiver(e.get_x(), e.get_y(), f_dic[str(e.get_id())][0], f_dic[str(e.get_id())][1], hatch='o',
                       width=0.01)
        # getting x,y for particles plot
        x = [e.get_x() for e in self.matrix.flatten()]
        y = [e.get_y() for e in self.matrix.flatten()]
        # plotting particles
        plt.scatter(x, y, c='r')
        # showing the plot
        plt.show()

    def plot_density(self, save=False, path=None, show=True, points=True):
        # plotting the density of the points
        plt.figure(figsize=(7, 7), dpi=80, facecolor='w', edgecolor='b')
        x = np.array([e.get_x() for e in self.matrix.flatten()])
        y = np.array([e.get_y() for e in self.matrix.flatten()])
        nbins = 300
        k = kde.gaussian_kde([x, y])
        xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        # plot a density
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap='viridis', shading='auto')
        if points:
            plt.scatter(x, y, c='r', alpha=0.1)
        # plt.colorbar()
        if save:
            plt.savefig(path, dpi=100)
        if show:
            plt.show()
        plt.close()
        plt.clf()

    def plot_density_3d(self, save=False, path=None, show=True):
        # plotting the density of the points in 3d
        fig = plt.figure(figsize=(7, 7), dpi=80, facecolor='w', edgecolor='b')
        ax = plt.axes(projection='3d')
        x = np.array([e.get_x() for e in self.matrix.flatten()])
        y = np.array([e.get_y() for e in self.matrix.flatten()])
        nbins = 100
        k = kde.gaussian_kde([x, y])
        xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        # plot a density
        ax.plot_surface(xi, yi, zi.reshape(xi.shape), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        if save:
            plt.savefig(path, dpi=100)
        if show:
            plt.show()
        plt.close()
        plt.clf()

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
    plate_neg = Plate_Negative(n=4, p1=[0, 0, 0], p2=[1, 1, 0], random=True)
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
    # moving the particle by a force vector
    # i = str(plate_neg.matrix.flatten()[1].get_id())
    # print(i)
    # plate_neg.move_by_force_vector(id=i, force=np.array([0.23, 0.2, 0]))
    # # checking via plot
    # plate_neg.plot_matrix_particles()
    # print(plate_neg.matrix.flatten()[0].get_x())
