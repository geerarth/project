import numpy as np

class Board():

    # here we place class attribute (static members)

    # init 
    def __init__(self, cycle=False):
        self.cycle = cycle
        self.squares = np.ones(15)
        self.position = 0
    
    # regular step
    def step(self, size):
        self.position += size

    # reset (in case of circular)
    def reset_position(self):
        if self.cycle == False:
            print(" /!\ Board isn't cyclic /!\ ")
        else:
            self.position = 0

    def get_position(self):
        return self.position

