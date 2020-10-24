# this file is simulating a simple grid model with two parallel capacitors
# loading in modules
import uuid
import math
import numpy as np
from scipy.constants import pi
# physical_constants[name] = (value, unit, uncertainty)
from scipy.constants import physical_constants as physical_constants


class Electron:
    # the class will handle the electrons and their movement
    def __init__(self, x, y, z):
        # setting the values for the single classes
        # setting coordinates
        self.__x = x
        self.__y = y
        self.__z = z
        # setting i-hat and ^j
        self._i = 1
        self._j = 1
        # setting charge
        self.__charge = -1 * physical_constants["elementary charge"][0]
        # setting id for identification
        self._id = uuid.uuid4()
        # setting factor k for force cal
        self.__k = 14 * pi * physical_constants['vacuum electric permittivity'][0]

    def cal_force(self, particle):
        # finding out the force between the two particles
        force = (self.__k * self.__charge * particle.__charge) / ((((particle.__x - self.__x) ** 2 + (
                particle.__y - self.__y) ** 2 + (particle.__z - self.__z) ** 2) ** 0.5) ** 2)
        # print("Force: ", force)
        # finding the vector for the force
        # setting unit vector for force
        unit_force_vector = np.array(
            [particle.__x - self.__x, particle.__y - self.__y, particle.__z - self.__z]) / np.linalg.norm(
            np.array([particle.__x - self.__x, particle.__y - self.__y, particle.__z - self.__z]))
        # print("Unit Vector force: ", unit_force_vector)
        # setting the force vector
        force_vector = unit_force_vector * force
        # print("Full force vector: ", force_vector)
        # getting angle for vx and vy cal
        # setting values
        force_vector_x = np.array([force_vector[0], 0, 0])
        force_vector_y = np.array([0, force_vector[1], 0])
        force_vector_z = np.array([0, 0, force_vector[2]])
        # returning all vectors
        return force, force_vector, force_vector_x, force_vector_y, force_vector_z

    def get_x(self):
        # getting x coordinate
        return self.__x

    def set_x(self, x):
        # setting the x coordinates
        self.__x = x

    def get_y(self):
        # getting y coordinate
        return self.__y

    def set_y(self, y):
        # setting the y coordinates
        self.__y = y

    def get_z(self):
        # getting z coordinate
        return self.__z

    def set_z(self, z):
        # setting the z coordinates
        self.__z = z

    def get_id(self):
        # getting the id from the electron
        return self._id

    def get_charge(self):
        # get the charge of the electron
        return self.__charge

    def __repr__(self):
        # printing name of class
        return "The class handles all actions of electrons"

    def __str__(self):
        # printing th object out for information
        return "This is electron: {0}, with the charge: {1}, and the coordinates, x: {2} and y: {3} and z: {4}".format(
            self._id, self.__charge, self.__x, self.__y, self.__z)

    # def __del__(self):
    #     # deleting function information
    #     print("Deleting electron: " + str(self._id) + " on coordinates, x: " + str(self.__x) + " y: " + str(self.__y))


class Plate_negative:
    # this is a class which represents one negative plate
    # this plate is filled with electrons which can move freely
    None


class Plate_positive:
    # this is a class which represents one positive plate
    # here are all the protons fixed anc can't move
    None


class Plate_Capacitor:
    # this capacitor represents two plates which interact together
    None


if __name__ == "__main__":
    # TODO split classes in diffrent files
    # TODO remoake Electrone to Particle in gernael
    # TODO make also z axis in Particle because 2D not enoiugh
    # TODO build postive plate with random and none random distribuation
    # printing class info
    print(Electron)
    # setting the first single electron
    e = Electron(x=5, y=2, z=8)
    # printing all information about it
    print(e)
    # getting values
    print("Coordinates: x: ", e.get_x(), "| y: ", e.get_y(), "| z: ", e.get_z())
    print("ID of electron: ", e.get_id(), " , charge: ", e.get_charge())
    # checking if cal works
    e2 = Electron(x=5.5, y=3, z=8)
    print(e.cal_force(particle=e2))
