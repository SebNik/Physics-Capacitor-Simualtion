# this file is working with two plate capacitors
import numpy as np
import matplotlib.pyplot as plt
from Plate_positive import Plate_Positive
from Plate_negative import Plate_Negative


class Plate_Capacitor:
    # this capacitor represents two plates which interact together
    def __init__(self, n, p1, p2, plane_z_pos, plane_z_neg, random):
        # setting the points
        self._p1 = p1
        self._p2 = p2
        # setting up a plane negative and positive
        self.plate_pos = Plate_Positive(n=n, p1=p1 + plane_z_pos, p2=p2 + plane_z_pos, random=random)
        self.plate_neg = Plate_Negative(n=n, p1=p1 + plane_z_neg, p2=p2 + plane_z_neg, random=random)

    def cal_forces(self):
        # this function is calculating all the forces for the particles
        # setting lists with force vectors and dic
        force_list = []
        force_dic = {}
        # getting inner forces
        inner_list, inner_dic = self.plate_neg.get_inner_forces()
        # getting forces for each electron with the positive charge and then adding it to inner forces
        for e_n in self.plate_neg.matrix.flatten():
            force_sum = np.array([0.0, 0.0, 0.0])
            # now going through positive plane
            for e_p in self.plate_pos.matrix.flatten():
                force, force_vector, force_vector_x, force_vector_y, force_vector_z = e_n.cal_force(particle=e_p)
                force_sum += force_vector
            # adding force_sum and inner force together
            force_sum += inner_dic[str(e_n.get_id())]
            # adding the force sum of all in list and dic
            force_list.append(force_sum)
            force_dic[str(e_n.get_id())] = force_sum
        # returning all vales
        return force_list, force_dic

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
    cap = Plate_Capacitor(n=3, p1=[0, 0], p2=[1e-15, 1e-15], plane_z_pos=[0], plane_z_neg=[1e-15], random=False)
    # cap.plate_neg.get_inner_forces()
    # plotting the room
    # cap.plotting_plates()
    # getting the forces for the particles
    print(cap.cal_forces())
    # print(cap.plate_neg.matrix[0][0].get_id())
    # plotting forces
    cap.plotting_plates_vectors_force()
