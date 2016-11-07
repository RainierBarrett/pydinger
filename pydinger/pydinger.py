import numpy as np
import numpy.polynomial.legendre as L

class Grid:
    '''This class is a grid implementation for holding our input data'''
    def __init__(self, axes, values):
        self.axes = axes
        self.values = values
        self.dim = len(axes)
        self.c = 1.0#this is the constant used in the Hamiltonian
        self.fourier = True#use Fourier series by default
        self.N = 50#a fairly accurate number

    def set_c(self, new_c):
        self.c = new_c
        
    def set_N(self, new_N):
        self.N = new_N

    def set_basis(self, new_bool):
        self.fourier = new_bool
        

def read_file(filename, dim = 1):#1D by default
    coords = [[] for i in range(dim)]#make the empty coordinates 'array'
    values = []
    with open(filename) as f:
        for line in f:
            words = line.split(' ')
            for i in range(dim):
                coords[i].append(float(words[i]))
            values.append(float(words[dim]))#this makes a big long potentials list for higher dims
    return(Grid(coords, values))

def read_input(filename = 'fourier_test_input.txt'):
    with open(filename) as f:
        for line in f:
            if('TARGET' in line):
                target = (line.split(' ')[1]).split('\n')[0]
                grid = read_file(target)
            elif('CONSTANT' in line):
                constant = float(line.split(' ')[1])
                grid.set_c(constant)
            elif('BASIS' in line):
                basis_bool = bool(int(line.split(' ')[1]))
                grid.set_basis(basis_bool)
            elif('SIZE' in line):
                size = int(line.split(' ')[1])
                grid.set_N(size)
    return(grid)
            #still will need to edit this later to also take in a wavefunction
