# this file is working with the negative plate
import uuid
import time
import numpy as np
import pandas as pd
from scipy.stats import kde
from Particle import Particle
import matplotlib.pyplot as plt
from scipy.constants import electron_mass


class Plate:
    # this is a class which represents one negative plate
    # this plate is filled with electrons which can move freely
    def __init__(self, n, p1, p2, type, random=False):
        # setting an id for the plate
        self._id = uuid.uuid4()
        # setting up the charge and type of the plate
        self.type = type
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
                data.append(Particle(x=x, y=y, z=self.z_plane, type_c=self.type))
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

    def move_plane_on_z_plane(self, new_z_plane):
        # this function will move the whole plate on the z plane
        self.z_plane = new_z_plane
        # moving all the particles
        for p in self.matrix.flatten():
            # changing the z axis
            p.set_z(z=new_z_plane)

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

    def find_relevant_part_of_particles(self):
        # this function is going to find the quarter of the particles that are necessary
        # first finding mirror axis x and y
        mirror_axis_x = self._x_length / 2 + self._p1[0]
        mirror_axis_y = self._y_length / 2 + self._p1[1]
        # searching for the relevant particles
        # a particle is relevant when its coordinates are smaller than the mirror axis
        relevant_particles_ids = []
        for p in self.matrix.flatten():
            if p.get_x() < mirror_axis_x and p.get_y() < mirror_axis_y:
                relevant_particles_ids.append(p.get_id())
        # returning the relevant data
        return relevant_particles_ids

    def find_corresponding_particles(self):
        # this function will find the mirrored particles and their corresponding relevant particle
        # first finding mirror axis x and y
        mirror_axis_x = self._x_length / 2 + self._p1[0]
        mirror_axis_y = self._y_length / 2 + self._p1[1]
        # getting the relevant particles
        relevant_particles_ids = self.find_relevant_part_of_particles()
        # setting the corresponding particles
        corresponding_particles = {}
        state_particles = {}
        for id_particle in relevant_particles_ids:
            # add id to dic
            corresponding_particles[id_particle] = []
            for p in self.matrix.flatten():
                if p.get_id() == id_particle:
                    # get coordinates for the particle
                    x, y = round(p.get_x(), 10), round(p.get_y(), 10)
                    # mirror first on the x and y axis
                    x_mirror = round((mirror_axis_x - x) + mirror_axis_x, 10)
                    y_mirror = round((mirror_axis_y - y) + mirror_axis_y, 10)
                    # checking for the mirrored particles
                    for p_test in self.matrix.flatten():
                        check = 'nee'
                        # X axis mirror
                        if round(p_test.get_x(), 10) == x_mirror and round(p_test.get_y(), 10) == y:
                            corresponding_particles[id_particle].append(p_test.get_id())
                            check = 'x'
                            state_particles[p_test.get_id()] = check
                        # Y axis mirror
                        if round(p_test.get_x(), 10) == x and round(p_test.get_y(), 10) == y_mirror:
                            corresponding_particles[id_particle].append(p_test.get_id())
                            check = 'y'
                            state_particles[p_test.get_id()] = check
                        # XY axis mirror
                        if round(p_test.get_x(), 10) == x_mirror and round(p_test.get_y(), 10) == y_mirror:
                            corresponding_particles[id_particle].append(p_test.get_id())
                            check = 'xy'
                            state_particles[p_test.get_id()] = check
        # returning the right value particles
        return corresponding_particles, state_particles

    def get_inner_forces_optimised(self, corresponding_particles_information=None):
        # getting the inner force of the plate faster
        # checking if the data is there or if it has to be found
        if corresponding_particles_information is None:
            corresponding_particles, state_particles = self.find_corresponding_particles()
        else:
            corresponding_particles, state_particles = corresponding_particles_information[0], \
                                                       corresponding_particles_information[1]
        # setting up the dic and list
        forces_list = []
        forces_dic = {}
        # iterating through electrons
        for e_cal in self.matrix.flatten():
            if e_cal.get_id() in corresponding_particles.keys():
                force_sum = np.array([0.0, 0.0, 0.0])
                for e_check in self.matrix.flatten():
                    if e_cal != e_check:
                        force, force_vector, force_vector_x, force_vector_y, force_vector_z = e_cal.cal_force(
                            particle=e_check)
                        force_sum += force_vector
                forces_list.append(force_sum)
                forces_dic[str(e_cal.get_id())] = force_sum
        # setting the other non relevant particles some forces vectors too
        for id_relevant in corresponding_particles:
            ids_non_relevant = corresponding_particles[id_relevant]
            original_force_vector = forces_dic[str(id_relevant)]
            for id in ids_non_relevant:
                state = state_particles[id]
                # setting the new force vector
                if state == 'x':
                    new_force_vector = original_force_vector * np.array([-1, 1, 1])
                elif state == 'y':
                    new_force_vector = original_force_vector * np.array([1, -1, 1])
                elif state == 'xy':
                    new_force_vector = original_force_vector * np.array([-1, -1, 1])
                forces_dic[str(id)] = new_force_vector
                forces_list.append(new_force_vector)
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

    def sumColumn(self, m):
        return [sum(col) for col in zip(*m)]

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

    def plot_matrix_particles(self, save=False, path=None, show=True, highlight=None):
        # plotting the particles
        plt.figure(figsize=(7, 7), dpi=80, facecolor='w', edgecolor='b')
        # getting x,y for particles plot
        x = [e.get_x() for e in self.matrix.flatten()]
        y = [e.get_y() for e in self.matrix.flatten()]
        color = []
        if type(highlight) == dict:
            color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'cyan', 'orangered']
            current_color = -1
            color_relevant_particles = {}
            for p in self.matrix.flatten():
                if p.get_id() in highlight.keys():
                    current_color += 1
                    if current_color == len(color_list):
                        current_color = 0
                    color_relevant_particles[p.get_id()] = color_list[current_color]
                    color.append(color_list[current_color])
                else:
                    state = True
                    for test_id in highlight:
                        if p.get_id() in highlight[test_id]:
                            color.append(color_relevant_particles[test_id])
                            state = False
                    if state:
                        color.append('w')
        else:
            for e in self.matrix.flatten():
                if highlight is not None:
                    if type(highlight) == list:
                        if e.get_id() in highlight:
                            color.append('g')
                        elif e.get_charge() > 0:
                            color.append('r')
                        else:
                            color.append('b')
                else:
                    if e.get_charge() > 0:
                        color.append('r')
                    else:
                        color.append('b')
        # plotting particles
        plt.scatter(x, y, c=color, alpha=0.5)
        if save:
            plt.savefig(path, dpi=100)
        if show:
            plt.show()
        plt.close()
        plt.clf()

    def plot_matrix_particles_vector_old(self):
        # plotting the particles and inner force vectors
        # setting figure
        plt.figure(figsize=(7, 7), dpi=80, facecolor='w', edgecolor='b')
        # getting forces data
        f_list, f_dic = self.get_inner_forces()
        print(f_list)
        # print(f_dic)
        f = 4e22
        for e in self.matrix.flatten():
            plt.quiver(e.get_x(), e.get_y(), f_dic[str(e.get_id())][0] * f, f_dic[str(e.get_id())][1] * f, hatch='o',
                       width=0.01, scale=1)
        # getting x,y for particles plot
        x = [e.get_x() for e in self.matrix.flatten()]
        y = [e.get_y() for e in self.matrix.flatten()]
        color = ['r' if e.get_charge() > 0 else 'b' for e in self.matrix.flatten()]
        # print(color)
        # plotting particles
        plt.scatter(x, y, c=color)
        # showing the plot
        plt.axis([-0.02, 0.05, -0.02, 0.05])
        plt.show()

    def plotting_every_single_force_vector(self):
        # plotting all of the force vectors
        # setting up colors
        color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'cyan']
        # assigning all the particles their color
        color_particle = {}
        c = 0
        for i in range(len(self.matrix.flatten())):
            color_particle[self.matrix.flatten()[i].get_id()] = color_list[c]
            c += 1
            if c > len(color_list) - 1:
                c = 0
        # plotting the particles
        # plt.figure(figsize=(7, 7), dpi=100, facecolor='w', edgecolor='b')
        # plt.axis([0, 0.03, 0, 0.03])
        # getting x,y for particles plot
        x = [e.get_x() for e in self.matrix.flatten()]
        y = [e.get_y() for e in self.matrix.flatten()]
        # iterating through the particles and plotting them
        for e in self.matrix.flatten():
            for i in range(1, 9):
                c = 1
                # plotting the single particle
                for e_n in self.matrix.flatten():
                    if e_n.get_id() != e.get_id():
                        force, force_vector, force_vector_x, force_vector_y, force_vector_z = e.cal_force(particle=e_n)
                        force_vector = force_vector * 4e21
                        print(force_vector)
                        plt.quiver(e.get_x(), e.get_y(), force_vector[0], force_vector[1],
                                   color=color_particle[e_n.get_id()], scale=1, width=0.01)
                    for e in self.matrix.flatten():
                        plt.scatter(e.get_x(), e.get_y(), c=color_particle[e.get_id()], s=100)
                    if c == i:
                        plt.show()
                        plt.figure(figsize=(7, 7), dpi=100, facecolor='w', edgecolor='b')
                        plt.axis([-0.005, 0.035, -0.005, 0.035])
                        break
        # # getting forces data
        # f_list, f_dic = self.get_inner_forces_optimised()
        # # print(f_dic)
        # for e in self.matrix.flatten():
        #     plt.quiver(e.get_x(), e.get_y(), f_dic[str(e.get_id())][0], f_dic[str(e.get_id())][1], hatch='o',
        #                width=0.01)
        plt.show()

    def plot_sigma(self, grid):
        # plotting the sigma data
        # getting the particles data
        x = np.array([e.get_x() for e in self.matrix.flatten()])
        y = np.array([e.get_y() for e in self.matrix.flatten()])
        # building the grid
        xi, yi = np.mgrid[x.min():x.max():grid * 1j, y.min():y.max():grid * 1j]
        print(xi, yi)
        # first finding mirror axis x and y
        mirror_axis_x = self._x_length / 2 + self._p1[0]
        mirror_axis_y = self._y_length / 2 + self._p1[1]
        # setting up the data matrix
        data_density = np.zeros((xi.shape[0] - 1, xi.shape[1] - 1))
        # iterating through the data
        for i in range(len(xi) - 1):
            for j in range(len(yi) - 1):
                for particle in self.matrix.flatten():
                    # modifying the x and y coordinates
                    # x coordinates
                    if particle.get_x() < mirror_axis_x:
                        x_check = particle.get_x() + (1 / 100000)
                    elif particle.get_x() > mirror_axis_x:
                        x_check = particle.get_x() - (1 / 100000)
                    else:
                        x_check = particle.get_x()
                    # y coordinates
                    if particle.get_y() < mirror_axis_y:
                        y_check = particle.get_y() + (1 / 100000)
                    elif particle.get_y() > mirror_axis_y:
                        y_check = particle.get_y() - (1 / 100000)
                    else:
                        y_check = particle.get_y()
                    if round(xi[i][j], 4) <= x_check <= round(xi[i + 1][j], 4) and round(yi[i][j],
                                                                                         4) <= y_check <= round(
                        yi[i][j + 1], 4):
                        data_density[i, j] += 1
        print(data_density)
        # adding the plot figures
        fig = plt.figure(figsize=(6, 5), dpi=100, facecolor='w', edgecolor='b')
        # setting up the plot fig
        ax = fig.add_subplot(1, 1, 1)
        # plotting the grid
        im = plt.imshow(data_density)
        # showing the color
        fig.colorbar(im)
        # showing the data
        plt.show()
        plt.close()
        plt.clf()

    def plot_matrix_particles_vector_optimised(self):
        # plotting the particles and inner force vectors
        # setting figure
        plt.figure(figsize=(7, 7), dpi=80, facecolor='w', edgecolor='b')
        # getting forces data
        f_list, f_dic = self.get_inner_forces_optimised()
        # print(f_dic)
        for e in self.matrix.flatten():
            plt.quiver(e.get_x(), e.get_y(), f_dic[str(e.get_id())][0], f_dic[str(e.get_id())][1], hatch='o',
                       width=0.01)
        # getting x,y for particles plot
        x = [e.get_x() for e in self.matrix.flatten()]
        y = [e.get_y() for e in self.matrix.flatten()]
        color = ['r' if e.get_charge() > 0 else 'b' for e in self.matrix.flatten()]
        # print(color)
        # plotting particles
        plt.scatter(x, y, c=color)
        # showing the plot
        plt.show()

    def plot_density_cals(self, nbins=300):
        # getting the calculations for density plot
        # getting the points
        x = np.array([e.get_x() for e in self.matrix.flatten()])
        y = np.array([e.get_y() for e in self.matrix.flatten()])
        # setting the cals for gaussian
        k = kde.gaussian_kde([x, y])
        # setting the data grid
        xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
        # combining all the data
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        # returning all the values
        return xi, yi, zi, x, y

    def plot_density_cals_fake(self, path):
        # this function will read in an excel file as an distributions function to the field line cals
        # reading in the pandas dataframe from file
        df = pd.read_excel(path, index_col=0)
        # returning the data from excel file
        return df['y'].to_numpy(), int(df['y'].to_numpy().shape[0])

    def plot_density_self_made(self, nbins_inside=100, searching_box=19, save=False, path=None, show=True, points=False):
        # this function is plotting the data
        # plotting the density of the points
        fig = plt.figure(dpi=100, facecolor='w', edgecolor='b')
        # setting up the plot fig
        ax = fig.add_subplot(1, 1, 1)
        # getting the data
        xi, yi, zi, x, y = self.plot_density_self_made_cals(nbins_inside=nbins_inside, searching_box=searching_box)
        print(xi)
        print(yi)
        # for i in range(len(xi)):
        #     for j in range(len(yi)):
        #         plt.hlines(y=yi[i, j], xmin=xi.min(), xmax=xi.max(), colors='black')
        #         plt.vlines(x=xi[i, j], ymin=yi.min(), ymax=yi.max(), colors='black')
        # plotting the grid
        im = plt.imshow(zi, extent=(self._p1[0], self._p2[0], self._p1[1], self._p2[1]))
        # showing the color
        fig.colorbar(im)
        if points:
            # getting x,y for particles plot
            x = [e.get_x() for e in self.matrix.flatten()]
            y = [e.get_y() for e in self.matrix.flatten()]
            color = ['r' if e.get_charge() > 0 else 'b' for e in self.matrix.flatten()]
            # plotting particles
            plt.scatter(x, y, c=color, alpha=0.5)
        if save:
            plt.savefig(path, dpi=100)
        if show:
            # plt.axis([0.005, 0.025, 0.005, 0.025])
            plt.show()
        plt.close()
        plt.clf()

    def plot_density_self_made_cals(self, nbins_inside=100, searching_box=19):
        # this is a density function for the plate it does the cals
        # getting the points
        x = np.array([e.get_x() for e in self.matrix.flatten()])
        y = np.array([e.get_y() for e in self.matrix.flatten()])
        # print(x)
        # print(y)
        # grid points outside
        outside_grid = (searching_box - 1) / 2
        # getting the delta
        delta = self.x_length / nbins_inside
        # grid inside the plate
        xi, yi = np.mgrid[x.min() - (outside_grid * delta):x.max() + (outside_grid * delta):(nbins_inside + 1 + (
                outside_grid * 2)) * 1j, y.min() - (outside_grid * delta):y.max() + (outside_grid * delta):(
                                                                                                                   nbins_inside + 1 + (
                                                                                                                   outside_grid * 2)) * 1j]
        # print(xi)
        # print(yi)
        # fig = plt.figure(dpi=300, facecolor='w', edgecolor='b')
        # ax = fig.add_subplot(1, 1, 1)
        # # plotting the grid
        # for i in range(len(xi)):
        #     for j in range(len(yi)):
        #         plt.hlines(y=yi[i, j], xmin=xi.min(), xmax=xi.max(), colors='black')
        #         plt.vlines(x=xi[i, j], ymin=yi.min(), ymax=yi.max(), colors='black')
        # # plotting the particles
        # # plt.scatter(xi, yi, c='r', alpha=0.5)
        # plt.scatter(x, y, c='b', s=100)
        # setting the density plate for the grid
        data_density_plate = np.zeros(
            (xi.shape[0] - 1 - int(outside_grid * 2), xi.shape[1] - 1 - int(outside_grid * 2)))
        outside_grid = int(outside_grid)
        print(outside_grid)
        # setting the density with zeros
        data_density = np.zeros((xi.shape[0] - 1, xi.shape[1] - 1))
        # iterating through the grid and rectangles
        for i in range(len(xi) - 1):
            for j in range(len(yi) - 1):
                for particle in self.matrix.flatten():
                    # first finding mirror axis x and y
                    mirror_axis_x = self._x_length / 2 + self._p1[0]
                    mirror_axis_y = self._y_length / 2 + self._p1[1]
                    # modifying the x and y coordinates
                    # x coordinates
                    if particle.get_x() < mirror_axis_x:
                        x_check = particle.get_x() + (1 / 100000)
                    elif particle.get_x() > mirror_axis_x:
                        x_check = particle.get_x() - (1 / 100000)
                    else:
                        x_check = particle.get_x()
                    # y coordinates
                    if particle.get_y() < mirror_axis_y:
                        y_check = particle.get_y() + (1 / 100000)
                    elif particle.get_y() > mirror_axis_y:
                        y_check = particle.get_y() - (1 / 100000)
                    else:
                        y_check = particle.get_y()
                    if round(xi[i][j], 4) <= x_check <= round(xi[i + 1][j], 4) and round(yi[i][j],
                                                                                         4) <= y_check <= round(
                        yi[i][j + 1], 4):
                        # print('Inside ', i, j, round(xi[i][j], 4), round(xi[i + 1][j], 4),
                        # round(yi[i][j], 4),round(yi[i][j + 1], 4), particle.get_x(), particle.get_y())
                        data_density[i, j] += 1
        # print(data_density)
        # iterating through the density on the plate
        for i in range(len(data_density_plate)):
            for j in range(len(data_density_plate)):
                # print(i,j,outside_grid)
                data_density_plate[i, j] += data_density[i + outside_grid, j + outside_grid]
                for a in range(1, int(outside_grid + 1)):
                    data_density_plate[i, j] += data_density[i + outside_grid + a, j + outside_grid]
                    data_density_plate[i, j] += data_density[i + outside_grid - a, j + outside_grid]
                    data_density_plate[i, j] += data_density[i + outside_grid, j + outside_grid + a]
                    data_density_plate[i, j] += data_density[i + outside_grid, j + outside_grid - a]
                    for t in range(1, int(outside_grid + 1)):
                        # print(a,t)
                        data_density_plate[i, j] += data_density[i + outside_grid + a, j + outside_grid + t]
                        data_density_plate[i, j] += data_density[i + outside_grid + a, j + outside_grid - t]
                        data_density_plate[i, j] += data_density[i + outside_grid - a, j + outside_grid - t]
                        data_density_plate[i, j] += data_density[i + outside_grid - a, j + outside_grid + t]

        # im = plt.imshow(data_density_plate, extent=(self._p1[0], self._p2[0], self._p1[1], self._p2[1]), alpha=0.5)
        # # showing the color
        # fig.colorbar(im)
        # plt.axis([xi.min() - (1 / 10000), yi.max() + (1 / 10000), xi.min() - (1 / 10000), yi.max() + (1 / 10000)])
        # plt.show()

        # setting the data grid
        xi, yi = np.mgrid[x.min():x.max():nbins_inside * 1j, y.min():y.max():nbins_inside * 1j]
        # returning the data
        return xi, yi, data_density_plate, x, y

    def plot_density_distribution(self, nbins=300):
        # plotting the distribution of density
        plt.figure(figsize=(8, 8), dpi=150, facecolor='w', edgecolor='b')
        # getting the data
        xi, yi, zi, x, y = self.plot_density_cals(nbins=nbins)
        # setting it the right way
        zi = zi.reshape(xi.shape)
        # combining the data
        list_dist = self.sumColumn(m=zi)
        # list_dist, nbins = self.plot_density_cals_fake(path="C:\\Users\\lordv\\Documents\\Data_Density_Plot.xlsx")
        # setting the title
        plt.title('Distribution of density on plate')
        # setting stuff
        plt.xlabel('x profile steps')
        plt.ylabel('added density')
        # plotting the plots
        plt.plot(list_dist, c='b', linewidth=2)
        plt.show()

    def plot_density(self, save=False, path=None, show=True, points=True, nbins=300):
        # plotting the density of the points
        plt.figure(figsize=(6, 5), dpi=100, facecolor='w', edgecolor='b')
        # getting the data
        xi, yi, zi, x, y = self.plot_density_cals(nbins=nbins)
        # plot a density
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap='viridis', shading='auto')
        plt.colorbar()
        if points:
            plt.scatter(x, y, c='r', alpha=0.1)
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
        return "This is a Plate : {0}, with a charge. The bounding box coordinates are: p1: {1}, p2: {2}, p3: {3}, p4: {4}, on the z-plane: {5}".format(
            self._id, self._p1, self._p2, self._p3, self._p4, self.z_plane)

    @property
    def x_length(self):
        return self._x_length

    @property
    def y_length(self):
        return self._y_length


if __name__ == "__main__":
    # getting class information
    # print(Plate)
    # setting instance of single plate
    plate_neg = Plate(n=6, p1=[0, 0, 0], p2=[1, 1, 0], random=False, type='-')
    # printing all information about it
    # print(plate_neg)
    # getting values
    # plate_neg.get_info_of_particles()
    # plotting the density of the points
    # plate_neg.plot_density()
    # getting the inner forces
    # print(np.array(plate_neg.get_inner_forces()[0]).mean())
    # plotting inner forces
    # plate_neg.plot_matrix_particles_vector_old()
    # plate_neg.plot_matrix_particles_vector_optimised()
    corresponding_particles, states = plate_neg.find_corresponding_particles()
    for i in corresponding_particles:
        print(i, corresponding_particles[i])
    print(states)
    plate_neg.plot_matrix_particles(highlight=corresponding_particles)
    # running the time check
    # start_time = time.time()
    # for i in range(100):
    #     # forces = plate_neg.get_inner_forces_optimised()
    #     forces = plate_neg.get_inner_forces()
    #     print(i, forces)
    # # 201sec vs. 517sec
    # print("--- %s seconds ---" % (time.time() - start_time))
