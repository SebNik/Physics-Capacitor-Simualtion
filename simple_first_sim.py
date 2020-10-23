# this file is simulating a simple grid model with two parallel capacitors
# loading in modules
from scipy.constants import pi, elementary_charge


class Electron:
    # the class will handle the electrons and their movement
    def __init__(self, x, y):
        # setting the values for the single classes
        # setting coordinates
        self.__x = x
        self.__y = y
        self.__charge = -1 * elementary_charge


if __name__ == "__main__":
    # setting the first single electron
    e = Electron(x=5, y=2)
    # printing all information about it
