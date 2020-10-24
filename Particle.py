# this file is simulating a simple grid model with two parallel capacitors
# loading in modules
import uuid
import math
import numpy as np
from scipy.constants import pi
# physical_constants[name] = (value, unit, uncertainty)
from scipy.constants import physical_constants as physical_constants


class Particle:
    # the class will handle the Particles and their movement
    def __init__(self, x, y, z, type_c):
        # setting the values for the single classes
        # setting coordinates
        self.__x = x
        self.__y = y
        self.__z = z
        # setting i-hat and ^j
        self._i = 1
        self._j = 1
        self.type = type_c
        # setting charge
        if self.type == '-':
            self.fac = -1
        elif self.type == '+':
            self.fac = 1
        self.__charge = self.fac * physical_constants["elementary charge"][0]
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
        # getting the id from the Particle
        return self._id

    def get_type(self):
        # getting the type from the Particle
        return self.type

    def get_charge(self):
        # get the charge of the Particle
        return self.__charge

    def __repr__(self):
        # printing name of class
        return "This is Particle: {0}, with the charge: {1}, and the coordinates, x: {2} and y: {3} and z: {4}".format(
            self._id, self.__charge, self.__x, self.__y, self.__z)

    def __str__(self):
        # printing th object out for information
        return "This is Particle: {0}, with the charge: {1}, and the coordinates, x: {2} and y: {3} and z: {4}".format(
            self._id, self.__charge, self.__x, self.__y, self.__z)

    # def __del__(self):
    #     # deleting function information
    #     print("Deleting Particle: " + str(self._id) + " on coordinates, x: " + str(self.__x) + " y: " + str(self.__y))


if __name__ == "__main__":
    # printing class info
    print(Particle)
    # setting the first single electron
    e = Particle(x=5, y=2, z=8, type_c='-')
    # printing all information about it
    print(e)
    # getting values
    print("Coordinates: x: ", e.get_x(), "| y: ", e.get_y(), "| z: ", e.get_z())
    print("ID of electron: ", e.get_id(), " , charge: ", e.get_charge(), " , type: ", e.get_type())
    # checking if cal works
    e2 = Particle(x=5.5, y=3, z=8, type_c='-')
    print(e.cal_force(particle=e2))
