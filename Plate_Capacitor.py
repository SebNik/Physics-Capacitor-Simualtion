# this file is working with two plate capacitors
import os
import json
import pickle
import datetime
import warnings
import numpy as np
from Plate import Plate
from Particle import Particle
import matplotlib.pyplot as plt
from scipy.constants import electron_mass
from scipy.constants import physical_constants as physical_constants

from os import listdir
from os.path import isfile, join

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


class Plate_Capacitor:
    # this capacitor represents two plates which interact together
    def __init__(self, n_neg, n_pos, p1, p2, plane_z_pos, plane_z_neg, random, name='Test'):
        # setting the points n
        self.rel_list = []
        self._n_neg = n_neg
        self._n_pos = n_pos
        # setting the points
        self._p1 = p1
        self._p2 = p2
        self.z_plane_diff = abs(plane_z_neg[0] - plane_z_pos[0])
        # setting up a plane negative and positive
        self.plate_pos = Plate(n=n_pos, p1=p1 + plane_z_pos, p2=p2 + plane_z_pos, type='+', random=random)
        self.plate_neg = Plate(n=n_neg, p1=p1 + plane_z_neg, p2=p2 + plane_z_neg, type='-', random=random)
        # setting the base path
        self.path = os.path.abspath(
            os.path.join('resources', 'exports', datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S") + '_' + name))
        # creating the base path
        os.mkdir(self.path)
        # setting dic
        d = {
            'n_neg': self._n_neg,
            'n_pos': self._n_pos,
            'p1': self._p1,
            'p2': self._p2,
            'plane_pos_z': plane_z_pos,
            'plane_neg_z': plane_z_neg,
            'name': name,
            'random': random
        }
        # dic settings in the file
        with open(self.path + "\\my_settings.json", "w") as f:
            json.dump(d, f)

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

    def cal_forces_optimised(self, corresponding_info_particles_neg=None, corresponding_state_info_particles_neg=None,
                             corresponding_info_particles_pos=None, corresponding_state_info_particles_pos=None):
        # this function is calculating all the forces for the particles faster
        # -------------------------- SET -------------------------
        # setting lists with force vectors and dic
        force_list_neg = []
        force_dic_neg = {}
        force_list_pos = []
        force_dic_pos = {}
        # get relevant corresponding information
        if corresponding_info_particles_neg is None:
            corresponding_info_particles_neg, corresponding_state_info_particles_neg = self.plate_neg.find_corresponding_particles()
            corresponding_info_particles_pos, corresponding_state_info_particles_pos = self.plate_pos.find_corresponding_particles()
        # -------------------------- INNER FORCES -------------------------
        # getting inner forces
        inner_list_neg, inner_dic_neg = self.plate_neg.get_inner_forces_optimised(
            corresponding_particles_information=[corresponding_info_particles_neg,
                                                 corresponding_state_info_particles_neg])
        # -------------------------- RELEVANT PLANE PARTICLES -------------------------
        # getting forces for each electron with the positive charge and then adding it to inner forces
        for e_n in self.plate_neg.matrix.flatten():
            if e_n.get_id() in corresponding_info_particles_neg.keys():
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
        # -------------------------- NON RELEVANT PLANE PARTICLES ------------------------------------
        # setting the other non relevant particles some forces vectors too
        new_force_vector = None
        for id_relevant in corresponding_info_particles_neg:
            ids_non_relevant = corresponding_info_particles_neg[id_relevant]
            original_force_vector = force_dic_neg[str(id_relevant)]
            for id in ids_non_relevant:
                state = corresponding_state_info_particles_neg[id]
                # setting the new force vector
                if state == 'x':
                    new_force_vector = original_force_vector * np.array([-1, 1, 1])
                elif state == 'y':
                    new_force_vector = original_force_vector * np.array([1, -1, 1])
                elif state == 'xy':
                    new_force_vector = original_force_vector * np.array([-1, -1, 1])
                force_dic_neg[str(id)] = new_force_vector
                force_list_neg.append(new_force_vector)
        # -------------------------- PLANE POSITIVE PARTICLES ----------------------------
        # setting the forces for the positive particles
        for e in self.plate_neg.matrix.flatten():
            new_force_vector = force_dic_neg[str(e.get_id())] * np.array([1, 1, -1])
            for e_p in self.plate_pos.matrix.flatten():
                if e.get_x() == e_p.get_x() and e.get_y() == e_p.get_y():
                    force_list_pos.append(new_force_vector)
                    force_dic_pos[str(e_p.get_id())] = new_force_vector
        # returning all values
        return force_list_neg, force_dic_neg, force_list_pos, force_dic_pos

    def same_position_of_particles(self, e1, e2):
        # this function is finding out if two particles have the same positions
        if e1.get_x() == e2.get_x() and e1.get_y() == e2.get_y() and e1.get_z() == e2.get_z():
            return False
        else:
            return True

    def cal_electric_field_3d(self, resolution_2d=5, resolution_3d=10, size=1):
        # this function is calculating the electric field between the two plates
        # setting the numpy spaces for the grid points
        delta = self.plate_neg.x_length * (size - 1) / 2
        x = np.linspace(0 - delta, self.plate_neg.x_length + delta, resolution_2d) + self._p1[0]
        y = np.linspace(0 - delta, self.plate_neg.y_length + delta, resolution_2d) + self._p1[1]
        z = np.linspace(0, self.z_plane_diff, resolution_3d) + self.plate_pos.z_plane
        # iterating through the whole cube of data
        # setting to new data lists
        array_results = []  # TODO build this to numpy
        forces_results = np.array([0, 0, 0, 0])
        for i in range(0, resolution_3d):
            # print out status
            print('Iteration on z plane res list: ', i)
            for j in range(0, resolution_2d):
                for k in range(0, resolution_2d):
                    # building mock particle
                    e_test = Particle(x=x[j], y=y[k], z=z[i], type_c='-')
                    # setting force sum vector
                    sum_forces = np.array([0.0, 0.0, 0.0])
                    # cal forces between test particle and all real ones
                    # negative plate
                    for e_n in self.plate_neg.matrix.flatten():
                        if self.same_position_of_particles(e1=e_n, e2=e_test):
                            force, force_vector, force_vector_x, force_vector_y, force_vector_z = e_test.cal_force(
                                particle=e_n)
                            sum_forces += force_vector
                    # positive plate
                    for e_p in self.plate_pos.matrix.flatten():
                        if self.same_position_of_particles(e1=e_p, e2=e_test):
                            force, force_vector, force_vector_x, force_vector_y, force_vector_z = e_test.cal_force(
                                particle=e_p)
                            sum_forces += force_vector
                    # building forces array
                    # forces_results.append([x[i], y[j], z[k], sum_forces])
                    forces_results = np.append(forces_results, [x[j], y[k], z[i], sum_forces], axis=0)
                    # cal the electric field on this point
                    e = sum_forces / physical_constants["elementary charge"][0]
                    array_results.append([x[j], y[k], z[i], e])
        # setting it to numpy for later
        forces_results = np.array(forces_results)
        # setting array to numpy and sorting it
        array_results = np.array(array_results)
        array_results = array_results[array_results[:, 2].argsort()]  # First sort doesn't need to be stable.
        array_results = array_results[array_results[:, 1].argsort(kind='mergesort')]
        array_results = array_results[array_results[:, 0].argsort(kind='mergesort')]
        # returning value
        return array_results, len(array_results), forces_results

    def cal_electric_field_2D(self, z_plane, resolution=10, size=1):
        # this function is calculating the electric field between the two plates on a plane
        # setting the numpy spaces for the grid points
        delta = self.plate_neg.x_length * (size - 1) / 2
        x = np.linspace(0 - delta, self.plate_neg.x_length + delta, resolution) + self._p1[0]
        y = np.linspace(0 - delta, self.plate_neg.y_length + delta, resolution) + self._p1[1]
        # iterating through the whole cube of data
        # setting to new data lists
        array_results = []
        forces_results = []
        for i in range(0, resolution):
            # print out status
            # print('Iteration on biggest list: ', i)
            for j in range(0, resolution):
                # building mock particle
                e_test = Particle(x=x[i], y=y[j], z=z_plane, type_c='+')
                # setting force sum vector
                sum_forces = np.array([0.0, 0.0, 0.0])
                sum_forces_num = 0
                # cal forces between test particle and all real ones
                # negative plate
                for e_n in self.plate_neg.matrix.flatten():
                    if self.same_position_of_particles(e1=e_n, e2=e_test):
                        force, force_vector, force_vector_x, force_vector_y, force_vector_z = e_test.cal_force(
                            particle=e_n)
                        sum_forces += force_vector
                        sum_forces_num += force
                    # else:
                    #     print('Found: ', e_test.get_x(), e_n.get_x(), e_test.get_y(), e_n.get_y(), e_test.get_z(),
                    #           e_n.get_z())
                # positive plate
                for e_p in self.plate_pos.matrix.flatten():
                    if self.same_position_of_particles(e1=e_p, e2=e_test):
                        force, force_vector, force_vector_x, force_vector_y, force_vector_z = e_test.cal_force(
                            particle=e_p)
                        sum_forces += force_vector
                        sum_forces_num += force
                    # else:
                    #     print('Found: ', e_test.get_x(), e_p.get_x(), e_test.get_y(), e_p.get_y(), e_test.get_z(),
                    #           e_p.get_z())
                # building forces array
                forces_results.append([x[i], y[j], z_plane, sum_forces])
                # cal the electric field on this point
                e = sum_forces_num / physical_constants["elementary charge"][0]
                array_results.append([x[i], y[j], z_plane, e])
        # setting it to numpy for later
        forces_results = np.array(forces_results)
        # setting array to numpy and sorting it
        array_results = np.array(array_results)
        array_results = array_results[array_results[:, 2].argsort()]  # First sort doesn't need to be stable.
        array_results = array_results[array_results[:, 1].argsort(kind='mergesort')]
        array_results = array_results[array_results[:, 0].argsort(kind='mergesort')]
        # returning value
        return array_results, len(array_results), forces_results

    def cal_electric_field_2D_self_made_density(self, nbins, searching_box, z_plane, resolution=10, size=1):
        # this function is calculating the electric field between the two plates on a plane
        # setting the numpy spaces for the grid points
        delta = self.plate_neg.x_length * (size - 1) / 2
        x_p_test = np.linspace(0 - delta, self.plate_neg.x_length + delta, resolution) + self._p1[0]
        y_p_test = np.linspace(0 - delta, self.plate_neg.y_length + delta, resolution) + self._p1[1]
        # getting the data for the density 2d plot for integral cal
        xi, yi, zi, x, y = self.plate_pos.plot_density_self_made_cals(nbins_inside=nbins,
                                                                      searching_box=searching_box)
        # setting it the right way
        zi = zi.reshape(xi.shape)
        # preparing the density data smaller
        zi_x = 8.85 * 10 ** -12 / zi.mean()
        zi_density = zi_x / zi
        # iterating through the whole cube of data
        # setting to new data lists
        array_results = []
        forces_results = []
        for i in range(0, resolution):
            for j in range(0, resolution):
                # setting force sum vector
                sum_forces = np.array([0.0, 0.0, 0.0])
                sum_forces_num = 0
                # building mock particle
                p_test = Particle(x=x_p_test[i], y=y_p_test[j], z=z_plane, type_c='+')
                for k in range(xi.shape[0]):
                    for m in range(xi.shape[1]):
                        # setting up the particle for the negative site
                        p_test_neg = Particle(x=xi[k, m], y=yi[k, m], z=self.plate_neg.z_plane, type_c='-')
                        p_test_neg.set_charge(charge=zi_density[k, m])
                        # setting up the particle for the positive site
                        p_test_pos = Particle(x=xi[k, m], y=yi[k, m], z=self.plate_pos.z_plane, type_c='+')
                        p_test_pos.set_charge(charge=zi_density[k, m])
                        # calculating the force between the particles
                        force, force_vector, force_vector_x, force_vector_y, force_vector_z = p_test.cal_force_q(
                            particle=p_test_neg)
                        sum_forces += force_vector
                        sum_forces_num += force
                        force, force_vector, force_vector_x, force_vector_y, force_vector_z = p_test.cal_force_q(
                            particle=p_test_pos)
                        sum_forces += force_vector
                        sum_forces_num += force
                        # cal forces between test particle and all real ones
                forces_results.append([x_p_test[i], y_p_test[j], z_plane, sum_forces])
                # cal the electric field on this point
                e = sum_forces_num / physical_constants["elementary charge"][0]
                array_results.append([x_p_test[i], y_p_test[j], z_plane, e])
        # setting it to numpy for later
        forces_results = np.array(forces_results)
        # setting array to numpy and sorting it
        array_results = np.array(array_results)
        array_results = array_results[array_results[:, 2].argsort()]  # First sort doesn't need to be stable.
        array_results = array_results[array_results[:, 1].argsort(kind='mergesort')]
        array_results = array_results[array_results[:, 0].argsort(kind='mergesort')]
        # returning value
        return array_results, len(array_results), forces_results, zi_density

    def cal_electric_field_2D_profile(self, x_plane, resolution_x, resolution_y, size=1):
        # this function is calculating the electric field between the two plates on a plane
        # setting the numpy spaces for the grid points
        delta = self.plate_neg.x_length * (size - 1) / 2
        y = np.linspace(0 - delta, self.plate_neg.x_length + delta, resolution_y) + self._p1[0]
        z = np.linspace(0, self.z_plane_diff, resolution_x) + self.plate_pos.z_plane
        # print(y)
        # print(z)
        # iterating through the whole cube of data
        # setting to new data lists
        array_results = []
        forces_results = []
        for i in range(0, resolution_y):
            # print out status
            # print('Iteration on biggest list: ', i)
            for j in range(0, resolution_x):
                # building mock particle
                e_test = Particle(x=x_plane, y=y[i], z=z[j], type_c='+')
                # setting force sum vector
                sum_forces = np.array([0.0, 0.0, 0.0])
                sum_forces_num = 0
                # cal forces between test particle and all real ones
                # negative plate
                for e_n in self.plate_neg.matrix.flatten():
                    if self.same_position_of_particles(e1=e_n, e2=e_test):
                        force, force_vector, force_vector_x, force_vector_y, force_vector_z = e_test.cal_force(
                            particle=e_n)
                        sum_forces += force_vector
                        sum_forces_num += force
                    # else:
                    #     print('Found: ', e_test.get_x(), e_n.get_x(), e_test.get_y(), e_n.get_y(), e_test.get_z(),
                    #           e_n.get_z())
                # positive plate
                for e_p in self.plate_pos.matrix.flatten():
                    if self.same_position_of_particles(e1=e_p, e2=e_test):
                        force, force_vector, force_vector_x, force_vector_y, force_vector_z = e_test.cal_force(
                            particle=e_p)
                        sum_forces += force_vector
                        sum_forces_num += force
                    # else:
                    #     print('Found: ', e_test.get_x(), e_p.get_x(), e_test.get_y(), e_p.get_y(), e_test.get_z(),
                    #           e_p.get_z())
                # building forces array
                forces_results.append([x_plane, y[i], z[j], sum_forces])
                # cal the electric field on this point
                e = sum_forces_num / physical_constants["elementary charge"][0]
                array_results.append([x_plane, y[i], z[j], e])
        # setting it to numpy for later
        forces_results = np.array(forces_results)
        # setting array to numpy and sorting it
        array_results = np.array(array_results)
        array_results = array_results[array_results[:, 2].argsort()]  # First sort doesn't need to be stable.
        array_results = array_results[array_results[:, 1].argsort(kind='mergesort')]
        array_results = array_results[array_results[:, 0].argsort(kind='mergesort')]
        # returning value
        return array_results, len(array_results), forces_results

    def analysis_2d_profile(self, resolution_x=10, resolution_y=50, show=False, x_plane=None, size=1, small_fraction=6,
                            images_n=10):
        # this function is going to cal the electric field and other parameters
        # creating the paths to save
        if x_plane is None:
            x_plane = [(self._p2[1] - self._p1[1]) / 2 + self._p1[1]]
        # print(x_plane)
        path_field_2d = os.path.abspath(os.path.join(self.path, 'E_Field_2D_Profile'))
        # create folder for saves
        if not os.path.isdir(path_field_2d):
            os.mkdir(path_field_2d)
        for x in x_plane:
            # getting the data
            array_results, length, forces_results = self.cal_electric_field_2D_profile(x_plane=x,
                                                                                       resolution_x=resolution_x,
                                                                                       size=size,
                                                                                       resolution_y=resolution_y)
            # print(array_results)
            # print(length)
            # print(array_results.shape)
            # plotting the test
            # print(array_results)
            # saving the arrays
            np.savez_compressed(self.path + '\\e_field_array_profile_2d.npz', array_results, chunksize=100)
            np.savez_compressed(self.path + '\\forces_array_profile_2d.npz', forces_results, chunksize=100)
            # building the plots
            # getting max and min for plots
            min_v = 0.0
            max_value = np.linspace(max(array_results[:, 3]) / small_fraction, max(array_results[:, 3]), images_n)
            # setting legend data
            delta = self.plate_neg.x_length * (size - 1) / 2
            # image getting the 2d
            image = array_results[:, 3].reshape(resolution_y, resolution_x)
            for e in max_value:
                # image plotting in 2d
                fig, ax = plt.subplots()
                # print([self.plate_pos.z_plane, self.plate_neg.z_plane, self._p1[1] - delta, self._p2[1] + delta])
                # m = ax.imshow(image, vmin=min_v, vmax=max_v, **{
                #     'extent': [self._p1[0] - delta, self._p2[0] + delta, self._p1[1] - delta, self._p2[1] + delta]})
                m = ax.imshow(image, vmin=min_v, vmax=e, **{
                    'extent': [self.plate_pos.z_plane, self.plate_neg.z_plane, self._p1[1] - delta,
                               self._p2[1] + delta]})
                fig.colorbar(m)
                plt.title(str(round(x, 5)))
                if show:
                    plt.show()
                plt.savefig(path_field_2d + '\\E_Field_2D_Profile_' + str(round(x, 5)) + '_Res_X_' + str(
                    resolution_x) + '_Res_Y_' + str(resolution_y) + 'V_MAX_' + str(round(e, 12)) + '.png',

                            dpi=500)
                # clearing out memory
                plt.close()
                plt.clf()

    def analysis(self, resolution_2d=10, resolution_3d=10, show=False, size=1):
        # this function is going to cal the electric field and other parameters
        # creating the paths to save
        path_field_3d = os.path.abspath(os.path.join(self.path, 'E_Field_3D'))
        path_field_2d = os.path.abspath(os.path.join(self.path, 'E_Field_2D'))
        # create folder for saves
        # create folder for saves
        if not os.path.isdir(path_field_2d):
            os.mkdir(path_field_2d)
        if not os.path.isdir(path_field_3d):
            os.mkdir(path_field_3d)
        # getting the data
        array_results, length, forces_results = self.cal_electric_field_3d(resolution_2d=resolution_2d,
                                                                           resolution_3d=resolution_3d, size=size)
        # print(array_results)
        print("Array result shape all 3d: ", array_results.shape)
        # saving the arrays
        np.savez_compressed(self.path + '\\e_field_array.npz', array_results, chunksize=100)
        np.savez_compressed(self.path + '\\forces_array.npz', forces_results, chunksize=100)
        # building the plots
        # setting up the x,y,z axis plots offsets
        delta = self.plate_neg.x_length * (size - 1) / 2
        x = np.linspace(0 - delta, self.plate_neg.x_length + delta, resolution_2d) + self._p1[0]
        y = np.linspace(0 - delta, self.plate_neg.y_length + delta, resolution_2d) + self._p1[1]
        z = np.linspace(0, self.z_plane_diff, resolution_3d) + self.plate_pos.z_plane
        # building the grid in the mesh
        x_plot_3d, y_plot_3d = np.meshgrid(x, y)
        # getting max and min for plots
        max_v = 0.15  # max(array_results[:, 3])
        min_v = 0.0  # min(array_results[:, 3])
        # iterating through it for plots
        for off in z:
            # data 2d plot creation and filter
            filter_array_2d = array_results[:, 2] == off
            data_2d_plot = array_results[filter_array_2d]
            print("2D Data Filtered: ", off, data_2d_plot.shape)
            # getting the real data from vector to scalar values
            data = np.array([((i[0] ** 2) + (i[1] ** 2) + (i[2] ** 2)) ** 0.5 for i in data_2d_plot[:, 3]])
            # setting the image data in the right format
            image = data.reshape(int(len(data) / int(resolution_2d)), int(resolution_2d))
            # plotting the 3d plot
            print("3D Plotting array (must match: ", x_plot_3d.shape, y_plot_3d.shape, image.shape)
            # creating the 3d plot surface
            fig = plt.figure(figsize=(7, 7), dpi=80, facecolor='w', edgecolor='b')
            ax = plt.axes(projection='3d')
            ax.plot_surface(x_plot_3d, y_plot_3d, image, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
            ax.view_init(elev=35.)
            plt.savefig(path_field_3d + '\\E_Field_3D_' + str(format(round(off, 5), '.5f')) + '_Res_2D_' + str(
                resolution_2d) + '.png', dpi=100)
            if show:
                plt.show()
            # clearing out memory
            plt.close()
            plt.clf()
            # setting legend data
            delta = self.plate_neg.x_length * (size - 1) / 2
            # image plotting in 2d
            fig, ax = plt.subplots()
            m = ax.imshow(image, vmin=min_v, vmax=max_v, **{
                'extent': [self._p1[0] - delta, self._p2[0] + delta, self._p1[1] - delta, self._p2[1] + delta]})
            fig.colorbar(m)
            # setting title with the offset and the check sum
            plt.title(str(round(off, 5)) + ' Check: ' + str(int(sum(sum(image)))))
            if show:
                plt.show()
            plt.savefig(path_field_2d + '\\E_Field_2D_' + str(format(round(off, 5), '.5f')) + '_Res_2D_' + str(
                resolution_2d) + '.png', dpi=100)
            # clearing out memory
            plt.close()
            plt.clf()

    def analysis_2D(self, nbins, searching_box, resolution=10, show=False, z_plane=None, size=1):
        # this function is going to cal the electric field and other parameters
        path_e_data_2d = os.path.abspath(os.path.join(self.path, 'E_Field_2D_DATA'))
        # create folder for saves
        if not os.path.isdir(path_e_data_2d):
            os.mkdir(path_e_data_2d)
        # creating the paths to save
        if z_plane is None:
            z_plane = [self.plate_neg.z_plane]
        path_field_2d = os.path.abspath(os.path.join(self.path, 'E_Field_2D'))
        # create folder for saves
        if not os.path.isdir(path_field_2d):
            os.mkdir(path_field_2d)
        for z in z_plane:
            # getting the data
            array_results, length, forces_results, zi_density = self.cal_electric_field_2D_self_made_density(
                nbins=nbins, searching_box=searching_box, z_plane=z, resolution=resolution, size=size)
            # plotting the test
            # print(array_results)
            # saving the arrays
            np.savez_compressed(path_e_data_2d + '\\e_field_array_z_' + str(z) + '.npz', array_results, chunksize=100)
            np.savez_compressed(path_e_data_2d + '\\forces_array_z_' + str(z) + '.npz', forces_results, chunksize=100)
            # saving all the data
            np.savez_compressed(path_e_data_2d + '\\zi_density_charge.npz', zi_density,
                                chunksize=100)
            # building the plots
            # setting legend data
            delta = self.plate_neg.x_length * (size - 1) / 2
            # image getting the 2d
            image = array_results[:, 3].reshape(int(len(array_results) / int(resolution)), int(resolution))
            # image plotting in 2d
            fig, ax = plt.subplots()
            m = ax.imshow(image, **{
                'extent': [self._p1[0] - delta, self._p2[0] + delta, self._p1[1] - delta, self._p2[1] + delta]})
            fig.colorbar(m)
            plt.title(str(round(z, 5)))
            if show:
                plt.show()
            plt.savefig(path_field_2d + '\\E_Field_2D_' + str(round(z, 5)) + '_Res_' + str(resolution) + '.png',
                        dpi=250)
            # clearing out memory
            plt.close()
            plt.clf()

    def set_self_path(self, path):
        # setting the base path
        self.path = path

    def sim(self, t=0.0000001, end_stop=1e-05):
        # this function is simulating the sates and stopping with stable state
        # creating the path for saving the data
        path_density_neg = os.path.abspath(os.path.join(self.path, 'Neg_Density'))
        path_particles_neg = os.path.abspath(os.path.join(self.path, 'Neg_Particles'))
        path_density_pos = os.path.abspath(os.path.join(self.path, 'Pos_Density'))
        path_particles_pos = os.path.abspath(os.path.join(self.path, 'Pos_Particles'))
        path_density_neg_3d = os.path.abspath(os.path.join(self.path, 'Neg_Density_3D'))
        path_stuff = os.path.abspath(os.path.join(self.path, 'Stuff'))
        # create folder for today
        os.mkdir(path_density_neg)
        os.mkdir(path_particles_neg)
        os.mkdir(path_density_pos)
        os.mkdir(path_particles_pos)
        os.mkdir(path_density_neg_3d)
        os.mkdir(path_stuff)
        # finding the corresponding particles
        corresponding_info_particles_neg, corresponding_state_info_particles_neg = self.plate_neg.find_corresponding_particles()
        corresponding_info_particles_pos, corresponding_state_info_particles_pos = self.plate_pos.find_corresponding_particles()
        # files for saving stuff
        path_stuff_export_sma = os.path.abspath(os.path.join(path_stuff, 'export_sma.csv'))
        # iterating through sim
        i = 0
        rel_avg_sum = [1, 1]
        sma_list = [1]
        while sma_list[-1] > end_stop:
            # getting the forces for all the particles
            force_list_neg, force_dic_neg, force_list_pos, force_dic_pos = self.cal_forces_optimised(
                corresponding_info_particles_neg=corresponding_info_particles_neg,
                corresponding_state_info_particles_neg=corresponding_state_info_particles_neg,
                corresponding_info_particles_pos=corresponding_info_particles_pos,
                corresponding_state_info_particles_pos=corresponding_state_info_particles_pos)
            # 20:52--21:03 (11min)-->133 (old) || 21:03--21:14 (11min)-->217 (new)
            # setting status sim to 0
            rel_avg_sum = []
            # moving all the particles by their force on the neg plate
            for e_n in self.plate_neg.matrix.flatten():
                # moving the particle
                s, x_rel, y_rel, rel_avg = self.plate_neg.move_by_force_time(id=str(e_n.get_id()),
                                                                             force=force_dic_neg[str(e_n.get_id())],
                                                                             delta_t=t)
                rel_avg_sum.append(rel_avg)
            # moving all the particles by their force on the pos plate
            for e_p in self.plate_pos.matrix.flatten():
                # moving the particle
                s, x_rel, y_rel, rel_avg = self.plate_pos.move_by_force_time(id=str(e_p.get_id()),
                                                                             force=force_dic_pos[str(e_p.get_id())],
                                                                             delta_t=t)
                rel_avg_sum.append(rel_avg)
            # setting indicators
            self.rel_list.append(abs(sum(rel_avg_sum) / len(rel_avg_sum)))
            i += 1
            # checking if every 10th sav image of plot
            if i % 5 == 0:
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
            # getting the SMA change
            n = 10
            sma_value = sum(np.array(self.rel_list)[-n:]) / n
            sma_list.append(sma_value)
            # print out
            print("OUTPUT: Iteration: ", i, ' electrons moved: ', abs(sum(rel_avg_sum) / len(rel_avg_sum)),
                  ' SMA n=10: ', sma_value)
        # exporting the sma
        np.savetxt(path_stuff_export_sma, np.array(sma_list), delimiter=";")
        # saving the class in pickle
        with open(self.path + '\\' + "class.pickle", "wb") as file_:
            pickle.dump(self, file_, -1)
        plt.plot(self.rel_list, label='Relative Sum Avg', c='r')
        plt.savefig(self.path + '\\sim.png', dpi=100)
        print('Sim done')

    def write_class_in_pickle(self):
        # this function saves the class into pickle
        with open(self.path + '\\' + "class.pickle", "wb") as file_:
            pickle.dump(self, file_, -1)

    def sumColumn(self, m):
        return [sum(col) for col in zip(*m)]

    def plot_field_lines(self, path=None, num_field_lines=10, delta_m=0.000004, x_plane=None, show=False, logs=True,
                         room=False):
        # this function is going to build the field lines for the plot
        # # setting up the path
        path_field_lines_2d = os.path.abspath(os.path.join(self.path, 'Field_Lines_2D'))
        path_field_lines_3d = os.path.abspath(os.path.join(self.path, 'Field_Lines_3D'))
        path_field_lines_2d_data = os.path.abspath(os.path.join(self.path, 'Field_Lines_2D_DATA'))
        # create folder for saves
        if not os.path.isdir(path_field_lines_2d):
            os.mkdir(path_field_lines_2d)
        if not os.path.isdir(path_field_lines_3d):
            os.mkdir(path_field_lines_3d)
        if not os.path.isdir(path_field_lines_2d_data):
            os.mkdir(path_field_lines_2d_data)
        # building the field lines
        field_lines = []
        plotted_lines = 0
        # delta to add it up on every iteration
        delta = np.array([0.0, self.plate_pos.y_length / num_field_lines, 0.0])
        # iteration over the different z planes
        for x_off in x_plane:
            # getting the start point on the bottom
            start_p = np.array([x_off, self._p1[1], self.plate_pos.z_plane])
            start_point_cal = start_p
            # iterating over length of plate and number of field lines
            for i in range(1, num_field_lines + 2):
                if logs:
                    print("Starting field line cal: ", start_point_cal)
                # setting the points data list for this one field line
                points_data = []
                # setting count for print out
                count = 0
                # setting up and test particle to find line
                p_test = Particle(x=start_point_cal[0], y=start_point_cal[1], z=start_point_cal[2], type_c='+')
                # p_test.set_charge_to_fraction(f=0.0001)
                # building the while loop for the stopping point
                while p_test.get_z() <= self.plate_neg.z_plane:
                    # setting force sum vector
                    sum_forces = np.array([0.0, 0.0, 0.0])
                    # cal forces between test particle and all real ones
                    # negative plate
                    for e_n in self.plate_neg.matrix.flatten():
                        if self.same_position_of_particles(e1=e_n, e2=p_test):
                            force, force_vector, force_vector_x, force_vector_y, force_vector_z = p_test.cal_force_q(
                                particle=e_n)
                            sum_forces += force_vector
                    # positive plate
                    for e_p in self.plate_pos.matrix.flatten():
                        if self.same_position_of_particles(e1=e_p, e2=p_test):
                            force, force_vector, force_vector_x, force_vector_y, force_vector_z = p_test.cal_force_q(
                                particle=e_p)
                            sum_forces += force_vector
                    # getting the unit vector
                    unit_vector = sum_forces / np.linalg.norm(sum_forces)
                    # getting the new vector
                    new_force_vector = unit_vector * delta_m
                    # setting new position
                    x_new = p_test.get_x() + new_force_vector[0]
                    y_new = p_test.get_y() + new_force_vector[1]
                    z_new = p_test.get_z() + new_force_vector[2]
                    # moving the particle by the new adjusted force vector
                    p_test.set_x(x=x_new)
                    p_test.set_y(y=y_new)
                    p_test.set_z(z=z_new)
                    # adding the new position and force in the list to the array
                    points_data.append([x_new, y_new, z_new, sum_forces])
                    # setting count higher
                    count += 1
                    if count % 100 == 0 and logs:
                        print('Count: ', count, 'Current position: ', p_test.get_x(), p_test.get_y(), p_test.get_z(),
                              ' distance to end: ', self.plate_neg.z_plane - p_test.get_z(), ' current force: ',
                              sum_forces)
                    # if self.plate_neg.z_plane - p_test.get_z() < self.z_plane_diff * 0.1:
                    #     break
                # setting the points data
                points_data = np.array(points_data)[1:-1]
                # saving the created data
                np.savez_compressed(
                    path_field_lines_2d_data + '\\e_field_lines_' + str(start_point_cal).replace(' ', '_').replace('.',
                                                                                                                   '_') + '.npz',
                    points_data, chunksize=100)
                # getting the forces for this particle and
                # adding the new filed line in big field lines
                field_lines.append(points_data)
                # plotting for the 2d line plot
                # plt.plot(points_data[:, 2], points_data[:, 1], c='darkgreen', linewidth=0.2)
                if plotted_lines != 0 and num_field_lines != plotted_lines:
                    # plotting for the 2d line plot
                    plt.plot(points_data[:, 2], points_data[:, 1], c='darkgreen', linewidth=0.2)
                    print(start_point_cal)
                plotted_lines += 1
                # setting the new start values for the next list
                start_point_cal = start_p + (delta * i)
            # building up the 2D plot
            x1, y1 = [self.plate_pos.z_plane, self.plate_pos.z_plane], [self._p1[1], self._p2[1]]
            plt.plot(x1, y1, c='r')
            x2, y2 = [self.plate_neg.z_plane, self.plate_neg.z_plane], [self._p1[1], self._p2[1]]
            plt.plot(x2, y2, c='b')
            # set the right title
            plt.title('Field Lines X_off: ' + str(round(x_off, 3)))
            # saving the image
            print(plt.axis())
            plt.savefig(path_field_lines_2d + '\\Field_Lines_X_off_' + str(round(x_off, 3)) + '_' + str(
                num_field_lines) + '.png', dpi=200)
            # showing the plot if requests
            if show:
                plt.show()
            plt.close()
        field_lines = np.array(field_lines)
        if room:
            #  building up the plot
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            # plotting the plates for better view
            r = [self._p1[0], self._p2[0]]
            x, y = np.meshgrid(r, r)
            # plotting the pos plate
            pos_z = np.full((2, 2), self.plate_pos.z_plane)
            ax.plot_surface(x, y, pos_z, alpha=0.5, color='r')
            # plotting the neg plate
            neg_z = np.full((2, 2), self.plate_neg.z_plane)
            ax.plot_surface(x, y, neg_z, alpha=0.5, color='b')
            # plotting the field lines
            for line in field_lines:
                ax.plot(line[:, 0], line[:, 1], line[:, 2])
            # setting labels
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            # set the right title
            plt.title('Field Lines 3D Plot')
            for ev in range(0, 60, 2):
                for ii in range(0, 180, 10):
                    ax.view_init(elev=ev, azim=ii)
                    # saving the image
                    plt.savefig(path_field_lines_3d + '\\Field_Lines_3D_' + str(ev) + '_' + str(ii) + '.png', dpi=100)
        # showing the big 3d plot
        if show:
            plt.show()
        # returning the values
        return field_lines

    def plot_field_lines_integral_calculation(self, num_field_lines=100, delta_m=0.000004, nbins=30, x_plane=None,
                                              show=False, logs=True, room=False, fake_dist=False, path_fake_dist=None):
        # this function is going to build the field lines for the plot
        # setting up the path
        path_field_lines_2d = os.path.abspath(os.path.join(self.path, 'Field_Lines_2D'))
        path_field_lines_2d_data = os.path.abspath(os.path.join(self.path, 'Field_Lines_2D_DATA'))
        path_field_lines_3d = os.path.abspath(os.path.join(self.path, 'Field_Lines_3D'))
        # create folder for saves
        if not os.path.isdir(path_field_lines_2d):
            os.mkdir(path_field_lines_2d)
        if not os.path.isdir(path_field_lines_2d_data):
            os.mkdir(path_field_lines_2d_data)
        if not os.path.isdir(path_field_lines_3d):
            os.mkdir(path_field_lines_3d)
        # building the field lines
        field_lines = []
        plotted_lines = 0
        # loading the fake dist in data_plot_density
        if fake_dist:
            data_plot_density, nbins = self.plate_neg.plot_density_cals_fake(path=path_fake_dist)
            # RESET delta for big space in which we check for the number density
            delta_n = np.array([self.plate_pos.y_length / nbins])
            print(nbins, data_plot_density)
        else:
            # delta for big space in which we check for the number density
            delta_n = np.array([self.plate_pos.y_length / nbins])
            # getting the data for the density 2d plot for integral cal
            xi, yi, zi, x, y = self.plate_pos.plot_density_cals(nbins=nbins)
            # setting it the right way
            zi = zi.reshape(xi.shape)
            # combing zi in one dimension array
            data_plot_density = self.sumColumn(m=zi)
        # getting the area array
        area_array = data_plot_density * delta_n
        area_array_sum = area_array.sum()
        # finding out how many field lines per delta_n
        num_field_lines_in_area = []
        for area in area_array:
            num_field_lines_in_area.append(int((area * num_field_lines) / area_array_sum) + 1)
        # testing the data
        print(num_field_lines_in_area)
        check_real_field_lines = sum(num_field_lines_in_area)
        print(check_real_field_lines)
        # setting the separated delta for all bins
        delta_bins = delta_n / num_field_lines_in_area
        # iteration over the different z planes
        for x_off in x_plane:
            # getting the start point on the bottom
            start_p = np.array([x_off, self._p1[1], self.plate_pos.z_plane])
            # now starting to set all the relevant points for sim in field lines
            start_points_list = [start_p]
            for o in range(0, nbins):
                points_in_sector = num_field_lines_in_area[o]
                delta_in_sector = delta_bins[o]
                for u in range(points_in_sector):
                    new_point = start_points_list[-1] + np.array([0.0, delta_in_sector, 0.0])
                    start_points_list.append(new_point)
            # iterating over length of plate and number of field lines
            for i in range(0, len(start_points_list)):
                start_point_cal = start_points_list[i]
                if logs:
                    print("Starting field line cal: ", start_point_cal)
                # setting the points data list for this one field line
                points_data = []
                # setting count for print out
                count = 0
                # setting up and test particle to find line
                p_test = Particle(x=start_point_cal[0], y=start_point_cal[1], z=start_point_cal[2], type_c='+')
                # p_test.set_charge_to_fraction(f=0.0001)
                # building the while loop for the stopping point
                while p_test.get_z() <= self.plate_neg.z_plane:
                    # setting force sum vector
                    sum_forces = np.array([0.0, 0.0, 0.0])
                    # cal forces between test particle and all real ones
                    # negative plate
                    for e_n in self.plate_neg.matrix.flatten():
                        if self.same_position_of_particles(e1=e_n, e2=p_test):
                            force, force_vector, force_vector_x, force_vector_y, force_vector_z = p_test.cal_force_q(
                                particle=e_n)
                            sum_forces += force_vector
                    # positive plate
                    for e_p in self.plate_pos.matrix.flatten():
                        if self.same_position_of_particles(e1=e_p, e2=p_test):
                            force, force_vector, force_vector_x, force_vector_y, force_vector_z = p_test.cal_force_q(
                                particle=e_p)
                            sum_forces += force_vector
                    # getting the unit vector
                    unit_vector = sum_forces / np.linalg.norm(sum_forces)
                    # getting the new vector
                    new_force_vector = unit_vector * delta_m
                    # setting new position
                    x_new = p_test.get_x() + new_force_vector[0]
                    y_new = p_test.get_y() + new_force_vector[1]
                    z_new = p_test.get_z() + new_force_vector[2]
                    # moving the particle by the new adjusted force vector
                    p_test.set_x(x=x_new)
                    p_test.set_y(y=y_new)
                    p_test.set_z(z=z_new)
                    # adding the new position and force in the list to the array
                    points_data.append([x_new, y_new, z_new, sum_forces])
                    # setting count higher
                    count += 1
                    if count % 100 == 0 and logs:
                        print('Count: ', count, 'Current position: ', p_test.get_x(), p_test.get_y(), p_test.get_z(),
                              ' distance to end: ', self.plate_neg.z_plane - p_test.get_z(), ' current force: ',
                              sum_forces)
                    # if self.plate_neg.z_plane - p_test.get_z() < self.z_plane_diff * 0.1:
                    #     break
                # setting the points data
                points_data = np.array(points_data)[1:-1]
                # saving the created data
                np.savez_compressed(
                    path_field_lines_2d_data + '\\e_field_lines_' + str(start_point_cal).replace(' ', '_').replace('.',
                                                                                                                   '_') + '.npz',
                    points_data, chunksize=100)
                # adding the new filed line in big field lines
                field_lines.append(points_data)
                if plotted_lines != 0 and check_real_field_lines != plotted_lines:
                    # plotting for the 2d line plot
                    plt.plot(points_data[:, 2], points_data[:, 1], c='darkgreen', linewidth=0.2)
                    print(start_point_cal)
                plotted_lines += 1
            # building up the 2D plot
            x1, y1 = [self.plate_pos.z_plane, self.plate_pos.z_plane], [self._p1[1], self._p2[1]]
            plt.plot(x1, y1, c='r')
            x2, y2 = [self.plate_neg.z_plane, self.plate_neg.z_plane], [self._p1[1], self._p2[1]]
            plt.plot(x2, y2, c='b')
            # set the right title
            plt.title('Field_Lines_X_off_' + str(round(x_off, 3)) + '_' + str(num_field_lines))
            # saving the image
            print(plt.axis())
            plt.savefig(path_field_lines_2d + '\\Field_Lines_X_off_' + str(round(x_off, 3)) + '_' + str(
                num_field_lines) + '.png', dpi=200)
            # showing the plot if requests
            if show:
                plt.show()
            plt.close()
        field_lines = np.array(field_lines)
        if room:
            #  building up the plot
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            # plotting the plates for better view
            r = [self._p1[0], self._p2[0]]
            x, y = np.meshgrid(r, r)
            # plotting the pos plate
            pos_z = np.full((2, 2), self.plate_pos.z_plane)
            ax.plot_surface(x, y, pos_z, alpha=0.5, color='r')
            # plotting the neg plate
            neg_z = np.full((2, 2), self.plate_neg.z_plane)
            ax.plot_surface(x, y, neg_z, alpha=0.5, color='b')
            # plotting the field lines
            for line in field_lines:
                ax.plot(line[:, 0], line[:, 1], line[:, 2])
            # setting labels
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            # set the right title
            plt.title('Field Lines 3D Plot')
            for ev in range(0, 60, 2):
                for ii in range(0, 180, 10):
                    ax.view_init(elev=ev, azim=ii)
                    # saving the image
                    plt.savefig(path_field_lines_3d + '\\Field_Lines_3D_' + str(ev) + '_' + str(ii) + '.png', dpi=100)
        # showing the big 3d plot
        if show:
            plt.show()
        # returning the values
        return field_lines

    def plot_field_lines_integral_calculation_flatten(self, num_field_lines=100, delta_m=0.000004, nbins=10,
                                                      x_plane=None, show=False, logs=True, room=False, fake_dist=False,
                                                      path_fake_dist=None, searching_box=5):
        # this function is going to build the field lines for the plot
        # setting up the path
        path_field_lines_2d = os.path.abspath(os.path.join(self.path, 'Field_Lines_2D'))
        path_field_lines_2d_data = os.path.abspath(os.path.join(self.path, 'Field_Lines_2D_DATA'))
        path_field_lines_3d = os.path.abspath(os.path.join(self.path, 'Field_Lines_3D'))
        # create folder for saves
        if not os.path.isdir(path_field_lines_2d):
            os.mkdir(path_field_lines_2d)
        if not os.path.isdir(path_field_lines_2d_data):
            os.mkdir(path_field_lines_2d_data)
        if not os.path.isdir(path_field_lines_3d):
            os.mkdir(path_field_lines_3d)
        # building the field lines
        field_lines = []
        plotted_lines = 0
        # # getting the density plot data
        # xi, yi, zi, x, y = self.plate_neg.plot_density_cals(nbins=nbins)
        # delta for big space in which we check for the number density
        delta_n = np.array([self.plate_pos.y_length / nbins])
        # getting the data for the density 2d plot for integral cal
        xi, yi, zi, x, y = self.plate_pos.plot_density_self_made_cals(nbins_inside=nbins,
                                                                      searching_box=searching_box)
        # setting it the right way
        zi = zi.reshape(xi.shape)
        # combing zi in one dimension array
        # data_plot_density = self.sumColumn(m=zi)
        data_plot_density = zi[:, int(len(xi) / 2)]
        print(data_plot_density)
        # preparing the density data smaller
        zi_x = 8.85 * 10 ** -12 / zi.mean()
        zi_density = zi_x / zi
        # saving all the data
        np.savez_compressed(path_field_lines_2d_data + '\\data_plot_density.npz', data_plot_density,
                            chunksize=100)
        np.savez_compressed(path_field_lines_2d_data + '\\zi_density_charge.npz', zi_density,
                            chunksize=100)
        # getting the area array
        # area_array = data_plot_density * delta_n
        area_array_sum = data_plot_density.sum()
        # finding out how many field lines per delta_n
        num_field_lines_in_area = []
        for area in data_plot_density:
            num_field_lines_in_area.append(int(round((area * num_field_lines) / area_array_sum, 0)))
        print(num_field_lines_in_area)
        # saving the field lines data
        np.savez_compressed(path_field_lines_2d_data + '\\num_field_lines_in_area.npz', num_field_lines_in_area,
                            chunksize=100)
        # plotting the graphics and saving it
        sectors_plot = [q for q in range(len(num_field_lines_in_area))]
        plt.scatter(sectors_plot, num_field_lines_in_area)
        plt.savefig(path_field_lines_2d + '\\lines_in_bins.png')
        plt.close()
        # testing the data
        check_real_field_lines = sum(num_field_lines_in_area)
        print(check_real_field_lines)
        # setting the separated delta for all bins
        delta_bins = delta_n / num_field_lines_in_area
        # iteration over the different z planes
        for x_off in x_plane:
            # getting the start point on the bottom
            start_p = np.array([x_off, self._p1[1] + (1 / 100000), self.plate_pos.z_plane])
            # now starting to set all the relevant points for sim in field lines
            start_points_list = [start_p]
            for o in range(0, nbins):
                points_in_sector = num_field_lines_in_area[o]
                delta_in_sector = delta_bins[o]
                for u in range(points_in_sector):
                    new_point = start_points_list[-1] + np.array([0.0, delta_in_sector, 0.0])
                    if new_point[1] >= self._p2[1]:
                        new_point[1] = self._p2[1] - (1 / 100000)
                    start_points_list.append(new_point)
            print(start_points_list)
            # plotting the points list and sector data
            # start_points_list_plot = np.array(start_points_list)
            # plt.scatter(start_points_list_plot[:, 1], start_points_list_plot[:, 2], c='r', s=100)
            # # sector split data
            # sector_x = np.linspace(self._p1[0], self._p2[0], nbins+1)
            # for i in sector_x:
            #     plt.vlines(x=i, ymin=-0.01, ymax=0.01)
            # plt.show()
            # saving all the data
            np.savez_compressed(path_field_lines_2d_data + '\\start_points_list.npz', start_points_list, chunksize=100)
            # iterating over length of plate and number of field lines
            for i in range(0, len(start_points_list)):
                start_point_cal = start_points_list[i]
                if logs:
                    print("Starting field line cal: ", start_point_cal)
                # setting the points data list for this one field line
                points_data = [[start_point_cal[0], start_point_cal[1], start_point_cal[2], np.array([0.0, 0.0, 0.0])]]
                # setting count for print out
                count = 0
                # setting up and test particle to find line
                p_test = Particle(x=start_point_cal[0], y=start_point_cal[1], z=start_point_cal[2], type_c='+')
                # p_test.set_charge_to_fraction(f=0.0001)
                # building the while loop for the stopping point
                while p_test.get_z() <= self.plate_neg.z_plane:
                    # setting force sum vector
                    sum_forces = np.array([0.0, 0.0, 0.0])
                    # iterating through all the density particles
                    for k in range(xi.shape[0]):
                        for m in range(xi.shape[1]):
                            # setting up the particle for the negative site
                            p_test_neg = Particle(x=xi[k, m], y=yi[k, m], z=self.plate_neg.z_plane, type_c='-')
                            p_test_neg.set_charge(charge=zi_density[k, m])
                            # setting up the particle for the positive site
                            p_test_pos = Particle(x=xi[k, m], y=yi[k, m], z=self.plate_pos.z_plane, type_c='+')
                            p_test_pos.set_charge(charge=zi_density[k, m])
                            # calculating the force between the particles
                            force, force_vector, force_vector_x, force_vector_y, force_vector_z = p_test.cal_force_q(
                                particle=p_test_neg)
                            sum_forces += force_vector
                            force, force_vector, force_vector_x, force_vector_y, force_vector_z = p_test.cal_force_q(
                                particle=p_test_pos)
                            sum_forces += force_vector
                    # getting the unit vector
                    unit_vector = sum_forces / np.linalg.norm(sum_forces)
                    # getting the new vector
                    new_force_vector = unit_vector * delta_m
                    # setting new position
                    x_new = p_test.get_x() + new_force_vector[0]
                    y_new = p_test.get_y() + new_force_vector[1]
                    z_new = p_test.get_z() + new_force_vector[2]
                    # moving the particle by the new djusted force vector
                    p_test.set_x(x=x_new)
                    p_test.set_y(y=y_new)
                    p_test.set_z(z=z_new)
                    # adding the new position and force in the list to the array
                    points_data.append([x_new, y_new, z_new, sum_forces])
                    # setting count higher
                    count += 1
                    if count % 100 == 0 and logs:
                        print('Count: ', count, 'Current position: ', p_test.get_x(), p_test.get_y(), p_test.get_z(),
                              ' distance to end: ', self.plate_neg.z_plane - p_test.get_z(), ' current force: ',
                              sum_forces)
                    # if self.plate_neg.z_plane - p_test.get_z() < self.z_plane_diff * 0.1:
                    #     break
                # setting the points data
                points_data = np.array(points_data)[:-1]
                # saving the created data
                np.savez_compressed(
                    path_field_lines_2d_data + '\\e_field_lines_' + str(start_point_cal).replace(' ', '_').replace('.',
                                                                                                                   '_') + '.npz',
                    points_data, chunksize=100)
                # adding the new filed line in big field lines
                field_lines.append(points_data)
                if plotted_lines != 0 and check_real_field_lines - 1 != plotted_lines:
                    # plotting for the 2d line plot
                    plt.plot(points_data[:, 2], points_data[:, 1], c='darkgreen', linewidth=0.2)
                plotted_lines += 1
            # building up the 2D plot
            x1, y1 = [self.plate_pos.z_plane, self.plate_pos.z_plane], [self._p1[1], self._p2[1]]
            plt.plot(x1, y1, c='r')
            x2, y2 = [self.plate_neg.z_plane, self.plate_neg.z_plane], [self._p1[1], self._p2[1]]
            plt.plot(x2, y2, c='b')
            # set the right title
            plt.title('Field Lines X_off: ' + str(round(x_off, 3)))
            # saving the image
            print(plt.axis())
            plt.savefig(path_field_lines_2d + '\\Field_Lines_X_off_' + str(round(x_off, 3)) + '.png', dpi=200)
            # showing the plot if requests
            if show:
                plt.show()
            plt.close()
        field_lines = np.array(field_lines)
        if room:
            #  building up the plot
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            # plotting the plates for better view
            r = [self._p1[0], self._p2[0]]
            x, y = np.meshgrid(r, r)
            # plotting the pos plate
            pos_z = np.full((2, 2), self.plate_pos.z_plane)
            ax.plot_surface(x, y, pos_z, alpha=0.5, color='r')
            # plotting the neg plate
            neg_z = np.full((2, 2), self.plate_neg.z_plane)
            ax.plot_surface(x, y, neg_z, alpha=0.5, color='b')
            # plotting the field lines
            for line in field_lines:
                ax.plot(line[:, 0], line[:, 1], line[:, 2])
            # setting labels
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            # set the right title
            plt.title('Field Lines 3D Plot')
            for ev in range(0, 60, 2):
                for ii in range(0, 180, 10):
                    ax.view_init(elev=ev, azim=ii)
                    # saving the image
                    plt.savefig(path_field_lines_3d + '\\Field_Lines_3D_' + str(ev) + '_' + str(ii) + '.png', dpi=100)
        # showing the big 3d plot
        if show:
            plt.show()
        # returning the values
        return field_lines

    def plot_field_lines_from_data(self, x_off_prefix='e_field_lines_[0_015'):
        # plotting the field lines from the saved data
        fig = plt.figure(figsize=(6, 5), dpi=200, facecolor='w', edgecolor='b')
        # finding the right path
        path_field_lines_2d_data = os.path.abspath(os.path.join(self.path, 'Field_Lines_2D_DATA'))
        # getting all the files
        onlyfiles = [f for f in listdir(path_field_lines_2d_data) if isfile(join(path_field_lines_2d_data, f))]
        # iterating through files and plotting them
        for file in onlyfiles:
            if x_off_prefix in file:
                # opening the data
                points_data = np.load(path_field_lines_2d_data + '\\' + file, allow_pickle=True)['arr_0']
                plt.plot(points_data[:, 2], points_data[:, 1], c='darkgreen', linewidth=0.5)
        # building up the 2D plot
        x1, y1 = [self.plate_pos.z_plane, self.plate_pos.z_plane], [self._p1[1], self._p2[1]]
        plt.plot(x1, y1, c='r')
        x2, y2 = [self.plate_neg.z_plane, self.plate_neg.z_plane], [self._p1[1], self._p2[1]]
        plt.plot(x2, y2, c='b')

        path_field_lines_2d = os.path.abspath(os.path.join(self.path, 'Field_Lines_2D'))
        plt.savefig(path_field_lines_2d + '\\Field_Lines_X_off_' + str(round(0.015, 3)) + '.png', dpi=400)

        # showing the plot
        plt.show()
        # # set the right title
        # plt.title('Field Lines X_off: ' + str(round(x_off, 3)))

    def plot_field_lines_static(self, num_field_lines=10):
        # this function will plot a fully static field
        fig = plt.figure(figsize=(6, 5), dpi=250, facecolor='w', edgecolor='b')
        # delta to add it up on every iteration
        delta = np.array([0.0, self.plate_pos.y_length / num_field_lines])
        # showing the fieldline
        start_p = np.array([self.plate_pos.z_plane, self._p1[1]])
        start_point_cal = start_p
        # iterating over length of plate and number of field lines
        for i in range(1, num_field_lines + 2):
            if i == num_field_lines:
                plt.plot(np.array([start_point_cal[0], self.plate_neg.z_plane]),
                         [start_point_cal[1], start_point_cal[1]],
                         c='darkgreen', linewidth=1, label='Field lines')
            else:
                plt.plot(np.array([start_point_cal[0], self.plate_neg.z_plane]),
                         [start_point_cal[1], start_point_cal[1]],
                         c='darkgreen', linewidth=1)
            # setting the new start values for the next list
            start_point_cal = start_p + (delta * i)
        # building up the 2D plot
        x1, y1 = [self.plate_pos.z_plane, self.plate_pos.z_plane], [self._p1[1], self._p2[1]]
        plt.plot(x1, y1, c='r', label="Positive Plate")
        x2, y2 = [self.plate_neg.z_plane, self.plate_neg.z_plane], [self._p1[1], self._p2[1]]
        plt.plot(x2, y2, c='b', label='Negative Plate')
        # set the right title
        # plt.title('Full static field')
        # plt.axis([0.00030000000000000003, 0.0157, 0.0019876531873855687, 0.028012346812614407])
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
            if e_p.get_type() == '+':
                color = 'r'
            else:
                color = 'b'
            ax.scatter3D(e_p.get_x(), e_p.get_y(), e_p.get_z(), c=color)
        # plotting points of neg plate
        for e_n in self.plate_neg.matrix.flatten():
            if e_n.get_type() == '+':
                color = 'r'
            else:
                color = 'b'
            ax.scatter3D(e_n.get_x(), e_n.get_y(), e_n.get_z(), c=color)
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
        plt.savefig('test.png', dpi=250)
        plt.show()

    @property
    def p1(self):
        return self._p1


if __name__ == "__main__":
    # setting up an instances for test
    cap = Plate_Capacitor(n_neg=10, n_pos=10, p1=[0.01, 0.01], p2=[0.02, 0.02], plane_z_pos=[0.001],
                          plane_z_neg=[0.004],
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
    # force_list_neg, force_dic_neg_old, force_list_pos, force_dic_pos_old = cap.cal_forces()
    force_list_neg, force_dic_neg, force_list_pos, force_dic_pos = cap.cal_forces_optimised()
    cap.sim()
    # building analysis
    # cap.analysis(resolution=100)
    # # plotting density to heck sim
    # cap.plate_neg.plot_matrix_particles()
    # cap.plate_neg.plot_density()
    # print(cap.cal_electric_field())
