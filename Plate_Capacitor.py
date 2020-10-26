# this file is working with two plate capacitors
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from Plate_positive import Plate_Positive
from Plate_negative import Plate_Negative


class Plate_Capacitor:
    # this capacitor represents two plates which interact together
    def __init__(self, n_neg, n_pos, p1, p2, plane_z_pos, plane_z_neg, random):
        # setting the points n
        self._n_neg = n_neg
        self._n_neg = n_pos
        # setting the points
        self._p1 = p1
        self._p2 = p2
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
        # create folder for today
        os.mkdir(path)
        os.mkdir(path_density_neg)
        os.mkdir(path_particles_neg)
        os.mkdir(path_density_pos)
        os.mkdir(path_particles_pos)
        # starting sim and setting status list for overview in moved particles in iterations
        s_list = []
        rel_list = []
        x_rel_list=[]
        y_rel_list = []
        # iterating through sim
        i = 0
        s_sum = 100
        while i > 800:
            # getting the forces for all the particles
            force_list_neg, force_dic_neg, force_list_pos, force_dic_pos = self.cal_forces()
            # setting status sim to 0
            s_sum = 0
            x_rel_sum, y_rel_sum, rel_avg_sum = [],[],[]
            # moving all the particles by their force on the neg plate
            for e_n in self.plate_neg.matrix.flatten():
                # moving the particle
                s, x_rel, y_rel, rel_avg = self.plate_neg.move_by_force_time(id=str(e_n.get_id()), force=force_dic_neg[str(e_n.get_id())],delta_t=2000)
                s_sum += s
                x_rel_sum.append(x_rel)
                y_rel_sum.append(y_rel)
                rel_avg_sum.append(rel_avg)
            # moving all the particles by their force on the pos plate
            for e_p in self.plate_pos.matrix.flatten():
                # moving the particle
                s, x_rel, y_rel, rel_avg = self.plate_pos.move_by_force_time(id=str(e_p.get_id()), force=force_dic_pos[str(e_p.get_id())],delta_t=2000)
                s_sum += s
                x_rel_sum.append(x_rel)
                y_rel_sum.append(y_rel)
                rel_avg_sum.append(rel_avg)
            # setting indicators
            rel_list.append(sum(rel_avg_sum)/len(rel_avg_sum))
            x_rel_list.append(sum(x_rel_sum)/len(x_rel_sum))
            y_rel_list.append(sum(y_rel_sum) / len(y_rel_sum))
            s_list.append(s_sum)
            i += 1
            # checking if every 10th sav image of plot
            if i % 10 == 0:
                # plotting particles and density and saving them
                self.plate_neg.plot_density(save=True, path=path_density_neg + '\\Plate_Neg_' + str(i) + '_Density.png',
                                            show=False, points=False)
                self.plate_neg.plot_matrix_particles(save=True, path=path_particles_neg + '\\Plate_Neg_' + str(
                    i) + '_Particles.png', show=False)
                self.plate_pos.plot_density(save=True, path=path_density_pos + '\\Plate_Pos_' + str(i) + '_Density.png',
                                            show=False, points=False)
                self.plate_pos.plot_matrix_particles(save=True, path=path_particles_pos + '\\Plate_Pos_' + str(
                    i) + '_Particles.png', show=False)
            # print out
            print("OUTPUT: Iteration: ", i, ' electrons moved: ', s_sum)
        plt.plot(rel_list)
        plt.plot(x_rel_list)
        plt.plot(y_rel_list)
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
    cap = Plate_Capacitor(n_neg=7, n_pos=5, p1=[0, 0], p2=[0.01, 0.01], plane_z_pos=[0], plane_z_neg=[0.001],
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
    cap.sim()
    # plotting density to heck sim
    cap.plate_neg.plot_matrix_particles()
    cap.plate_neg.plot_density()
