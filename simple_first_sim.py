# this file is simulating a simple grid model with two parallel capacitors
# loading in modules
import uuid
from scipy.constants import pi, elementary_charge


class Electron:
    # the class will handle the electrons and their movement
    def __init__(self, x, y):
        # setting the values for the single classes
        # setting coordinates
        self.__x = x
        self.__y = y
        # setting charge
        self.__charge = -1 * elementary_charge
        # setting id for identification
        self._id = uuid.uuid4()

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

