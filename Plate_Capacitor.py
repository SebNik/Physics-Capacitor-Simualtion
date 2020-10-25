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
    cap = Plate_Capacitor(n=3, p1=[0, 0], p2=[0.001, 0.001], plane_z_pos=[0], plane_z_neg=[0.001], random=False)
    # cap.plate_neg.get_inner_forces()
    # plotting the room
    cap.plotting_plates()
