from pydinger import *

grid = read_input('./input.txt')
grid.do_variation(10000)
print("The final coefficients were: {}".format(grid.coefficients))
print("And that makes the ground state energy: {}".format(grid.get_energy()))
