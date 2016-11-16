import numpy as np
import numpy.polynomial.legendre as L

class Grid:
    '''This class is a grid implementation for holding our input data'''
    def __init__(self, axis):
        self.coefficients = []#holds our basis set coefficients later
        self.axis = np.array(axis)
        self.v0 = 0
        self.c = 1.0#this is the constant used in the Hamiltonian
        self.fourier = True#use Fourier series by default
        self.N = 50#a fairly accurate number
        self.period = abs(self.axis[0] - self.axis[len(self.axis)-1])#treat as if it's periodic
        self.hmat = []
        self.wavefunc = []

    def set_c(self, new_c):
        '''For setting the new constant in the operator.'''
        self.c = new_c
        
    def set_N(self, new_N):
        '''For setting how big our basis set will be.'''
        self.N = new_N

    def set_v(self, new_v):
        '''For setting our potential.'''
        self.v = new_v
        
    def set_basis(self, new_bool):
        '''For choosing which basis set to use.'''
        self.fourier = new_bool

    def set_wavefunc(self, func):
        '''Takes in a function of x as a python-formatted string. For example, "2*x + np.cos(x*np.pi)" would work. Evaluates the function on the axis of the grid and stores it as a numpy array.'''
        for x in self.axis:
            self.wavefunc.append(eval(func))
        self.wavefunc = np.array(self.wavefunc)

    def get_coefficients(self, func):
        '''This will dispatch to the appropriate coefficients getter function based on which basis set we're using.'''
        if(self.fourier == True):
            self.get_fourier_coefficients( func )
        elif(self.fourier == False):
            self.get_legendre_coefficients( func )

    def get_legendre_coefficients(self, func):
        '''This returns the actual Legendre-polynomial coefficient values for our function, up to N.'''
        self.coefficients = L.legfit(self.axis, func, self.N -1)

    def cn(self, func, n):
        '''This calculates the nth Fourier coefficient, using a function represented by a numpy array called 'func'. This is a Riemann sum approximation of the integral we would use for the inner product.'''
        c = func * np.exp(-1j*2*n*np.pi*self.axis/self.period)
        return(c.sum()/c.size)
    
    def f(self, func, x):
        '''This finds the actual fourier values based on the coefficients. Mostly for testing and personal peace of mind. Don't think this works, actually, if there are complex coefficients.'''
        bounds = np.arange(0.5, self.N + 0.5)
        coeff_vals = np.array([2*self.cn(func, i)*np.exp(1j*2*i*np.pi*x/self.period) for i in bounds])
        return(coeff_vals.sum())
    
    def get_fourier_coefficients(self, func):
        '''This fills an array with the N Fourier coefficient values. Only in 1D for now.'''
        for i in range(self.N):
            self.coefficients.append(self.cn(func, i))
        self.coefficients = np.array(self.coefficients)

    def get_values(self, func):
        '''This dispatches to the correct get_values function based on which basis we choose.'''
        if(self.fourier == True):
            return(self.get_fourier_values( func ))
        elif(self.fourier == False):
            return(self.get_legendre_values())
            
    def get_fourier_values(self, func):
        '''This will return a numpy array of values at each point on an axis corresponding to our Fourier coefficients found with get_fourier_coefficients().'''
        values = np.array([self.f(func, x).real for x in self.axis])
        return(values)

    def get_legendre_values(self):
        '''This returns a numpy array of values at each point on the axis corresponding to our Legendre polynomial values found with get_legendre_coefficients().'''
        values = np.array(L.legval(self.axis, self.coefficients))
        return(values)

    def get_hmat(self):
        '''This dispatches the appropriate hamiltonian for the basis set.'''
        if(self.fourier == True):
            self.get_hmat_fourier()

    def get_hmat_fourier(self):
        '''This constructs the Hamiltonian matrix for the Fourier basis set. Conveniently diagonal due to the nature of the Fourier series. Should only ever be called once per run.'''
        for i in range(self.N):
            self.hmat.append([0 for i in range(self.N)])
        for i in range(self.N):
            self.hmat[i][i] = (-4* (i**2) * ((np.pi)**2) / self.period)
            self.hmat = np.array(self.hmat)



    def apply_H(self):
        '''This applies the Hamiltonian operator, dispatching to the appropriate system.'''
        if(self.fourier == True):
            self.apply_H_fourier()
        elif(self.fourier == False):
            self.apply_H_legendre()

    def apply_H_legendre(self):
        '''This applies the Hamiltonian operator, utilizing a builtin capability of the numpy.polynomial.legendre module to get the second derivatives. Note that we have to "pad" the coefficients array with two zeros after taking the second derivative.'''
        #taking del^2 has never been easier!
        new_coefficients = L.legder(self.coefficients, 2)
        new_coefficients = list(new_coefficients)
        for i in range(2):
            new_coefficients.append(0)
        new_coefficients = np.array(new_coefficients)#what a pain!
        self.coefficients = new_coefficients*(-self.c) + self.v * self.coefficients
        

    def apply_H_fourier(self):
        '''This applies the Hamiltonian operator to our coefficient list in the Fourier basis.'''
        #this does the matrix multiplication we need:
        new_coefficients = np.dot(self.hmat, self.coefficients)
        #due to the way I have stored my hamiltonian matrix, I have to do the V adding here:
        self.coefficients = (new_coefficients*(-self.c) + self.v*self.coefficients*2*self.period)
            
def read_file(filename):
    '''This reads in a file containing the x-axis for our wavefunction.'''
    coords = [] 
    with open(filename) as f:
        for line in f:
            words = line.split(' ')
            coords.append(float(words[0]))
    return(Grid(coords))

def read_input(filename = 'fourier_test_input.txt'):
    '''This reads an input file with our chosen formatting.'''
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
            elif('POTENTIAL' in line):
                pot = float(line.split(' ')[1])
                grid.set_v(pot)
            elif('FUNCTION' in line):
                func = line.split("'")[1]
                grid.set_wavefunc(func)
    return(grid)
            #still will need to edit this later to also take in a wavefunction
