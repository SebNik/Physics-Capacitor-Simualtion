# this file is simulating a simple grid model with two parallel capacitors
# loading in modules
import uuid
import numpy as np
from scipy.constants import pi
# physical_constants[name] = (value, unit, uncertainty)
from scipy.constants import physical_constants as physical_constants


class Electron:
    # the class will handle the electrons and their movement
    def __init__(self, x, y):
        # setting the values for the single classes
        # setting coordinates
        self.__x = x
        self.__y = y
        # setting charge
        self.__charge = -1 * physical_constants["elementary charge"][0]
        # setting id for identification
        self._id = uuid.uuid4()
        # setting factor k for force cal
        self.__k = 14 * pi * physical_constants['vacuum electric permittivity'][0]

    def convert_vector_degrees(self, vector):
        # converting vector  into degrees by calculating degrees in radians from horizontal vector
        horizontal_vector = np.array([0, 1])
        vector = np.array(vector)
        # calculating a vector for unit
        unit_vector_horizontal = horizontal_vector / np.linalg.norm(horizontal_vector)
        unit_vector = vector / np.linalg.norm(vector)
        dot_product = np.dot(unit_vector_horizontal, unit_vector)
        # finding angles in vectors
        angle = np.arccos(dot_product)
        # returning angle
        return angle

    def cal_force(self):

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
        return "This is electron: {0}, with the charge: {1}, and the coordinates, x: {2} and y: {3}".format(self._id,
                                                                                                            self.__charge,
                                                                                                            self.__x,
                                                                                                            self.__y)

    def __del__(self):
        # deleting function information
        print("Deleting electron: " + str(self._id) + " on coordinates, x: " + str(self.__x) + " y: " + str(self.__y))


if __name__ == "__main__":
    # printing class info
    print(Electron)
    # setting the first single electron
    e = Electron(x=5, y=2)
    # printing all information about it
    print(e)
    # getting values
    print("Coordinates: x: ", e.get_x(), "| y: ", e.get_y())
    print("ID of electron: ", e.get_id(), " , charge: ", e.get_charge())
