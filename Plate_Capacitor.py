# this file is working with two plate capacitors
import os
import pickle
import datetime
import numpy as np
from Particle import Particle
import matplotlib.pyplot as plt
from Plate_negative import Plate_Negative
from scipy.constants import physical_constants as physical_constants


class Plate_Capacitor:
    # this capacitor represents two plates which interact together
    def __init__(self, n_neg, n_pos, p1, p2, plane_z_pos, plane_z_neg, random):
        # setting the points n
        self.rel_list = []
        self._n_neg = n_neg
        self._n_neg = n_pos
        # setting the points
        self._p1 = p1
        self._p2 = p2
        self.z_plane_diff = abs(plane_z_neg[0] - plane_z_pos[0])
        # setting up a plane negative and positive
        self.plate_pos = Plate_Negative(n=n_pos, p1=p1 + plane_z_pos, p2=p2 + plane_z_pos, random=random)
        self.plate_neg = Plate_Negative(n=n_neg, p1=p1 + plane_z_neg, p2=p2 + plane_z_neg, random=random)

    def cal_forces(self):
        # this function is calculating all the forces for the particles
        # setting lists with force vectors and dic
        force_list_neg = []
        force_dic_neg = {}
        force_list_pos = []
        force_dic_pos = {}
        # getting inner forces
        inner_list_neg, inner_dic_neg = self.plate_neg.get_inner_forces()
        inner_list_pos, inner_dic_pos = self.plate_pos.get_inner_forces()
        # -------------------------- Negative Plane -------------------------
        # getting forces for each electron with the positive charge and then adding it to inner forces
        for e_n in self.plate_neg.matrix.flatten():
            force_sum_neg = np.array([0.0, 0.0, 0.0])
            # now going through positive plane
            for e_p in self.plate_pos.matrix.flatten():
                force, force_vector, force_vector_x, force_vector_y, force_vector_z = e_n.cal_force(particle=e_p)
                force_sum_neg += force_vector
            # adding force_sum and inner force together
            force_sum_neg += inner_dic_neg[str(e_n.get_id())]
            # adding the force sum of all in list and dic
            force_list_neg.append(force_sum_neg)
            force_dic_neg[str(e_n.get_id())] = force_sum_neg
        # -------------------------- Positive Plane -------------------------
        # getting forces for each electron with the positive charge and then adding it to inner forces
        for e_n in self.plate_pos.matrix.flatten():
            force_sum_pos = np.array([0.0, 0.0, 0.0])
            # now going through positive plane
            for e_p in self.plate_neg.matrix.flatten():
                force, force_vector, force_vector_x, force_vector_y, force_vector_z = e_n.cal_force(particle=e_p)
                force_sum_pos += force_vector
            # adding force_sum and inner force together
            force_sum_pos += inner_dic_pos[str(e_n.get_id())]
            # adding the force sum of all in list and dic
            force_list_pos.append(force_sum_pos)
            force_dic_pos[str(e_n.get_id())] = force_sum_pos
        # returning all vales
        return force_list_neg, force_dic_neg, force_list_pos, force_dic_pos

    def cal_electric_field(self, resolution=40):
        # this function is calculating the electric field between the two plates
        # print(self.z_plane_diff, self.plate_neg.x_length, self.plate_neg.y_length)
        # setting the numpy spaces for the grid points
        x = np.linspace(0, self.plate_neg.x_length, resolution) + self._p1[0]
        y = np.linspace(0, self.plate_neg.y_length, resolution) + self._p1[1]
        z = np.linspace(0, self.z_plane_diff, resolution) + self.plate_pos.z_plane
        # print(x, '\n', y, '\n', z)
        # iterating through simple small cube with a 1/4 of the real volume
        # later building the cube up to full size
        array_results = []
        forces_results = []
        for i in range(0, resolution):
            for j in range(0, resolution):
                for k in range(0, resolution):
                    # building mock particle
                    e_test = Particle(x=x[i], y=y[j], z=z[k], type_c='-')
                    # setting force sum vector
                    sum_forces = np.array([0.0, 0.0, 0.0])
                    # cal forces between test particle and all real ones
                    # negative plate
                    for e_n in self.plate_neg.matrix.flatten():
                        if e_n.get_x() != e_test.get_x() and e_n.get_y() != e_test.get_y() and e_n.get_z() != e_test.get_z():
                            force, force_vector, force_vector_x, force_vector_y, force_vector_z = e_test.cal_force(
                                particle=e_n)
                            sum_forces += force_vector
                        else:
                            print('Skipping in cal particle:', e_n.get_x(), e_test.get_x(), e_n.get_y(), e_test.get_y(),
                                  e_n.get_z(), e_test.get_z())
                    # positive plate
                    for e_p in self.plate_pos.matrix.flatten():
                        if e_p.get_x() != e_test.get_x() and e_p.get_y() != e_test.get_y() and e_p.get_z() != e_test.get_z():
                            force, force_vector, force_vector_x, force_vector_y, force_vector_z = e_test.cal_force(
                                particle=e_p)
                            sum_forces += force_vector
                    # building forces array
                    forces_results.append([x[i], y[j], z[k], sum_forces])
                    # cal the electric field on this point
                    e = (sum_forces[0] ** 2 + sum_forces[1] ** 2 + sum_forces[2] ** 2) ** 0.5 / \
                        physical_constants["elementary charge"][0]
                    array_results.append([x[i], y[j], z[k], e])
        # setting array to numpy and sorting it
        array_results = np.array(array_results)
        array_results = array_results[array_results[:, 2].argsort()]  # First sort doesn't need to be stable.
        array_results = array_results[array_results[:, 1].argsort(kind='mergesort')]
        array_results = array_results[array_results[:, 0].argsort(kind='mergesort')]
        # print(array_results)
        data_2d_plot = []
        for r in array_results:
            if r[2] == 0.001:
                data_2d_plot.append(r)
        data_2d_plot = np.array(data_2d_plot)
        a = data_2d_plot[data_2d_plot[:, 2].argsort()]  # First sort doesn't need to be stable.
        a = a[a[:, 1].argsort(kind='mergesort')]
        a = a[a[:, 0].argsort(kind='mergesort')]
        # print(a, len(a))
        # plt.scatter(a[:, 0], a[:, 1], c=[a[:, 3]])
        # plt.colorbar()
        # plt.show()
        # image = a[:, 3].reshape(int(len(a) / int(resolution)), int(resolution))
        # fig, ax = plt.subplots()
        # ax.imshow(image)
        # plt.show()
        # returning value
        return array_results, len(array_results)

    def sim(self):
        # this function is simulating the sates and stopping with stable state
        # creating the path for saving the data
        path = os.path.abspath(
            os.path.join('resources', 'exports', datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S")))
        path_density_neg = os.path.abspath(
            os.path.join('resources', 'exports', datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S"), 'Neg_Density'))
        path_particles_neg = os.path.abspath(
            os.path.join('resources', 'exports', datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S"),
                         'Neg_Particles'))
        path_density_pos = os.path.abspath(
            os.path.join('resources', 'exports', datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S"), 'Pos_Density'))
        path_particles_pos = os.path.abspath(
            os.path.join('resources', 'exports', datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S"),
                         'Pos_Particles'))
        path_density_neg_3d = os.path.abspath(
            os.path.join('resources', 'exports', datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S"),
                         'Neg_Density_3D'))
        # create folder for today
        os.mkdir(path)
        os.mkdir(path_density_neg)
        os.mkdir(path_particles_neg)
        os.mkdir(path_density_pos)
        os.mkdir(path_particles_pos)
        os.mkdir(path_density_neg_3d)
        # iterating through sim
        i = 0
        rel_avg_sum = [1, 1]
        while abs(sum(rel_avg_sum) / len(rel_avg_sum)) > 0.0001:
            # getting the forces for all the particles
            force_list_neg, force_dic_neg, force_list_pos, force_dic_pos = self.cal_forces()
            # setting status sim to 0
            rel_avg_sum = []
            # moving all the particles by their force on the neg plate
            for e_n in self.plate_neg.matrix.flatten():
                # moving the particle
                s, x_rel, y_rel, rel_avg = self.plate_neg.move_by_force_time(id=str(e_n.get_id()),
                                                                             force=force_dic_neg[str(e_n.get_id())],
                                                                             delta_t=0.0000001)
                rel_avg_sum.append(rel_avg)
            # moving all the particles by their force on the pos plate
            for e_p in self.plate_pos.matrix.flatten():
                # moving the particle
                s, x_rel, y_rel, rel_avg = self.plate_pos.move_by_force_time(id=str(e_p.get_id()),
                                                                             force=force_dic_pos[str(e_p.get_id())],
                                                                             delta_t=0.0000001)
                rel_avg_sum.append(rel_avg)
            # setting indicators
            self.rel_list.append(sum(rel_avg_sum) / len(rel_avg_sum))
            i += 1
            # checking if every 10th sav image of plot
            if i % 10 == 0:
                # plotting particles and density and saving them
                self.plate_neg.plot_density_3d(save=True,
                                               path=path_density_neg_3d + '\\Plate_Neg_' + str(i) + '_3D_Density.png',
                                               show=False)
                self.plate_neg.plot_density(save=True, path=path_density_neg + '\\Plate_Neg_' + str(i) + '_Density.png',
                                            show=False, points=False)
                self.plate_neg.plot_matrix_particles(save=True, path=path_particles_neg + '\\Plate_Neg_' + str(
                    i) + '_Particles.png', show=False)
                self.plate_pos.plot_density(save=True, path=path_density_pos + '\\Plate_Pos_' + str(i) + '_Density.png',
                                            show=False, points=False)
                self.plate_pos.plot_matrix_particles(save=True, path=path_particles_pos + '\\Plate_Pos_' + str(
                    i) + '_Particles.png', show=False)
            # print out
            print("OUTPUT: Iteration: ", i, ' electrons moved: ', abs(sum(rel_avg_sum) / len(rel_avg_sum)))
        with open(path + '\\' + "class.pickle", "wb") as file_:
            pickle.dump(self, file_, -1)
        plt.plot(self.rel_list, label='Relative Sum Avg', c='r')
        plt.show()

    def plotting_plates_vectors_force(self):
        # plotting the 3D room of the electrons and their vectors
        # getting force vectors
        force_l, force_dic = self.cal_forces()
        # setting up space
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        # plotting points of pos plate
        for e_p in self.plate_pos.matrix.flatten():
            ax.scatter3D(e_p.get_x(), e_p.get_y(), e_p.get_z(), c='r')
        # plotting points of neg plate
        for e_n in self.plate_neg.matrix.flatten():
            ax.scatter3D(e_n.get_x(), e_n.get_y(), e_n.get_z(), c='b')
            ax.quiver(e_n.get_x(), e_n.get_y(), e_n.get_z(), force_dic[str(e_n.get_id())][0],
                      force_dic[str(e_n.get_id())][1], force_dic[str(e_n.get_id())][2])
        # plotting the plates for better view
        r = [self._p1[0], self._p2[0]]
        x, y = np.meshgrid(r, r)
        # plotting the pos plate
        pos_z = np.full((2, 2), self.plate_pos.z_plane)
        ax.plot_surface(x, y, pos_z, alpha=0.5, color='r')
        # plotting the neg plate
        neg_z = np.full((2, 2), self.plate_neg.z_plane)
        ax.plot_surface(x, y, neg_z, alpha=0.5, color='b')
        # setting labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # plotting out the room
        plt.show()

    def plotting_plates(self):
        # plotting the 3D room of the electrons
        # setting up space
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        # plotting points of pos plate
        for e_p in self.plate_pos.matrix.flatten():
            ax.scatter3D(e_p.get_x(), e_p.get_y(), e_p.get_z(), c='r')
        # plotting points of neg plate
        for e_n in self.plate_neg.matrix.flatten():
            ax.scatter3D(e_n.get_x(), e_n.get_y(), e_n.get_z(), c='b')
        # plotting the plates for better view
        r = [self._p1[0], self._p2[0]]
        x, y = np.meshgrid(r, r)
        # plotting the pos plate
        pos_z = np.full((2, 2), self.plate_pos.z_plane)
        ax.plot_surface(x, y, pos_z, alpha=0.5, color='r')
        # plotting the neg plate
        neg_z = np.full((2, 2), self.plate_neg.z_plane)
        ax.plot_surface(x, y, neg_z, alpha=0.5, color='b')
        # setting labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # plotting out the room
        plt.show()


if __name__ == "__main__":
    # setting up an instances for test
    cap = Plate_Capacitor(n_neg=14, n_pos=10, p1=[0.01, 0.01], p2=[0.02, 0.02], plane_z_pos=[0.001],
                          plane_z_neg=[0.002],
                          random=False)
    # plotting the room
    # cap.plotting_plates()
    # getting the forces for the particles
    # print(cap.cal_forces())
    # print(cap.plate_neg.matrix[0][0].get_id())
    # plotting forces
    # cap.plate_neg.plot_density()
    # cap.plotting_plates_vectors_force()
    # finding p
    # print(cap.find_p())
    # cap.plate_neg.plot_matrix_particles()
    # cap.plate_neg.plot_density()
    # starting sim
    # cap.sim()
    # # plotting density to heck sim
    cap.plate_neg.plot_matrix_particles()
    cap.plate_neg.plot_density()
    # print(cap.cal_electric_field())
