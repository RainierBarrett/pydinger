import numpy as np
import numpy.polynomial.legendre as L

class Grid:
    '''This class is a grid implementation for holding our input data'''
    def __init__(self, axis, values):
        self.coefficients = []#holds our basis set coefficients later
        self.axis = np.array(axis)
        self.values = np.array(values)
        self.c = 1.0#this is the constant used in the Hamiltonian
        self.fourier = True#use Fourier series by default
        self.N = 50#a fairly accurate number
        self.period = abs(self.axis[0] - self.axis[len(self.axis)-1])#treat as if it's periodic
        self.offset = self.axis[0]#the offset for plotting to use

    def set_c(self, new_c):
        '''For setting the new constant in the operator.'''
        self.c = new_c
        
    def set_N(self, new_N):
        '''For setting how big our basis set will be.'''
        self.N = new_N

    def set_basis(self, new_bool):
        '''For choosing which basis set to use.'''
        self.fourier = new_bool

    def get_coefficients(self, func):
        '''This will dispatch to the appropriate coefficients getter function based on which basis set we're using.'''
        if(self.fourier == True):
            self.get_fourier_coefficients( func )

    def cn(self, func, n):
        '''This calculates the nth Fourier coefficient, using a function represented by a numpy array called 'func'. This is a Riemann sum approximation of the integral we would use, and proves pretty accurate.'''
        c = func * np.exp(-1j*2*n*np.pi*func/self.period)
        return(c.sum()/c.size)
    
    def f(self, func, x):
        bounds = np.arange(0.5, self.N + 0.5)
        coeff_vals = np.array([2*self.cn(func, i)*np.exp(1j*2*i*np.pi*x/self.period) for i in bounds])
        return(coeff_vals.sum())
    
    def get_fourier_coefficients(self, func):
        '''This fills an array with the N Fourier coefficient values. Only in 1D for now.'''
        for i in range(self.N):
            self.coefficients.append(self.cn(func, i))

    def get_fourier_values(self, func):
        '''This will return a numpy array of values at each point on an axis corresponding to our Fourier coefficients found with get_fourier_coefficients().'''
        values = np.array([self.f(func, x).real for x in self.axis])
        return(values)

def read_file(filename, dim = 1):#1D by default
    coords = [] 
    values = []
    with open(filename) as f:
        for line in f:
            words = line.split(' ')
            coords.append(float(words[0]))
            values.append(float(words[1]))#only doing 1D
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
