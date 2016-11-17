#!/usr/bin/env python

"""
test_pydinger
----------------------------------

Tests for `pydinger` module.
"""


from __future__ import division
import sys
import unittest
from contextlib import contextmanager
from click.testing import CliRunner
import numpy as np
import copy

from pydinger import pydinger
from pydinger import cli

class TestPydinger(unittest.TestCase):

    def test_command_line_interface(self):
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'pydinger.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output

    def test_read_table_1D(self):
        '''This tests to make sure we can read in a 1D potential file correctly.'''
        testfile = '1D_test.txt'
        grid = pydinger.read_file(testfile)
        assert len(grid.axis) == 200
        assert grid.axis[0] + 0.9999 < 0.000001
        assert grid.axis[199] - 0.99 < 0.000001
        assert grid.period - 2.0  < 0.000001

    def test_read_input(self):
        '''This tests to make sure we can read an input file and fetch the appropriate potential file to go with it.'''
        testfile = 'fourier_test_input.txt'
        grid = pydinger.read_input(testfile)
        #check that we read the grid OK
        assert len(grid.axis) == 200
        assert grid.axis[0] + 0.9999 < 0.000001
        assert grid.axis[199] - 0.99 < 0.000001
        #print(grid.values[0])
        #check that we have the right values for our constants
        assert grid.c - 1.0 < 0.000001
        assert grid.N == 100
        assert grid.v - 2.0 < 0.000001
        assert grid.fourier == True
        for i in range(len(grid.axis)):
            assert (grid.wavefunc[i] - (grid.axis[i]**4 - grid.axis[i]**2) < 0.000001)
        testfile = 'legendre_test_input.txt'
        grid = pydinger.read_input(testfile)
        assert len(grid.axis) == 200
        assert grid.axis[0] + 0.9999 < 0.000001
        assert grid.axis[199] - 0.99 < 0.000001
        #check that we have the right values for our constants
        assert grid.c - 1.5 < 0.000001
        assert grid.N == 10
        assert grid.fourier == False
        for i in range(len(grid.axis)):
            assert (grid.wavefunc[i] - (grid.axis[i]**4 - grid.axis[i]**2) < 0.000001)
        #now check to make sure we can properly read input that has no function given
        testfile = 'fourier_no_function_input.txt'
        grid = pydinger.read_input(testfile)
        assert len(grid.axis) == 200
        assert grid.axis[0] + 0.9999 < 0.000001
        assert grid.axis[199] - 0.99 < 0.000001
        #print(grid.values[0])
        #check that we have the right values for our constants
        print("GRID's CONSTANT IS {}".format(grid.c))
        assert (grid.c - 1.0) < 0.00001
        assert grid.N == 100
        assert grid.v - 2.0 <  0.00001
        assert grid.fourier == True
        for item in grid.coefficients:
            assert (item - 1.0) < 0.00001
        

            
    def test_get_coeffs_fourier(self):
        '''This tests that we are able to recover a well-behaved function's Fourier polynomial coefficients. Seeing whether they're 'good' is beyond the scope of unit testing, but I made sure to play around with them in a Jupyter Notebook, and it works well.'''
        testfile = 'fourier_test_input.txt'
        grid = pydinger.read_input(testfile)
        #this is a nice-looking, well-behaved (easy to converge upon) function for a fourier series
        test_wavefunc = grid.axis**4 - grid.axis**2
        grid.get_coefficients(test_wavefunc)#should calculate the fourier coefficients
        assert len(grid.coefficients) == grid.N
        values = grid.get_values(test_wavefunc)
        diffsquare = 0
        for i in range(len(values)):
            diffsquare +=(values[i] - test_wavefunc[i])**2
        assert(diffsquare < 0.05)#that should be pretty accurate

    def test_get_coeffs_legendre(self):
        '''This will test if we are able to get the same function as used in the Fourier test by using Legendre polynomials instead, and properly plot from said coefficients. Like the Fourier test, how good this fits is a bit outside the scope of unit testing, but I've looked at the plots here in a Jupyter Notebook and it also works well.'''
        testfile = 'legendre_test_input.txt'
        grid = pydinger.read_input(testfile)
        test_wavefunc = grid.axis**4 - grid.axis**2
        grid.get_coefficients(test_wavefunc)
        assert len(grid.coefficients) == grid.N
        values = grid.get_values( test_wavefunc )
        assert len(values) == len(grid.axis)
        diffsquare = 0
        for i in range(len(values)):
            diffsquare +=(values[i] - test_wavefunc[i])**2
        assert(diffsquare < 0.05)#that should be pretty accurate


    def test_get_hmat_fourier(self):
        '''This tests to make sure we can obtain the Hamiltonian matrix from our Fourier basis representation of a wavefunction. Specifically, that the matrix is diagonal, which occurs naturally due to the orthonormality of this basis set.'''
        testfile = 'fourier_test_input.txt'
        grid = pydinger.read_input(testfile)
        test_wavefunc = np.sin(grid.axis)
        grid.get_coefficients(test_wavefunc)#now we have the original coefficients
        grid.get_hmat()#this creates the hamiltonian matrix
        assert len(grid.hmat) == grid.N
        assert len(grid.hmat[0]) == grid.N
        #the matrix will be diagonal, but the (0,0) entry will be 0 here, so:
        for i in range(1,grid.N):
            assert grid.hmat[i][i] != 0

        
    def test_apply_H_fourier(self):
        '''This will test to see if we can apply the Hamiltonian (in matrix representation) to a wavefunction with the Fourier series basis set.'''
        testfile = 'fourier_test_input.txt'
        grid = pydinger.read_input(testfile)
        test_wavefunc = np.sin(grid.axis)
        grid.get_coefficients(test_wavefunc)
        grid.get_hmat()
        old_coeffs = copy.deepcopy(grid.coefficients)
        new_coeffs = grid.apply_H()#now we should have operated the Hamiltonian
        for i in range(grid.N):#these should all change now, with our wavefunction
            assert new_coeffs[i] != old_coeffs[i]

    def test_apply_H_legendre(self):
        '''This tests that we can apply the Hamiltonian to a wavefunction with the Legendre polynomial basis set.'''
        testfile = 'legendre_test_input.txt'
        grid = pydinger.read_input(testfile)
        x = grid.axis #because the input file was from 0 to 2, this will be from -1 to 1
        test_wavefunc = x**2 -2*x**3 + 4
        grid.get_coefficients(test_wavefunc)
        #since the legendre polynomial module in numpy has a built-in method to just GET the second derivative basis set coefficients, I just use that.
        #note that we have to pad the new list it makes, because two of the coefficients will become zero when taking the second derivative, naturally...
        old_coeffs = copy.deepcopy(grid.coefficients)
        new_coeffs = grid.apply_H()
        for i in range(grid.N):#these should all change now, with our wavefunction
            assert new_coeffs[i] != old_coeffs[i]

    def test_get_energy_fourier(self):
        '''This tests to make sure we can evaluate an energy value given a set of Fourier series basis set coefficients. The energy value should be the same now matter how many times we take it, and should change if we change any of the basis set coefficients.'''
        test_axis = [i/200.0 -1 for i in range(401)]
        grid = pydinger.Grid(test_axis, True)
        grid.set_N(50)
        grid.set_v(1.0)
        grid.set_c(1.0)
        grid.get_hmat()
        energy = grid.get_energy()
        e2 = grid.get_energy()
        assert isinstance(energy, float)
        assert energy - e2 < 0.00001

    def test_get_energy_legendre(self):
        '''This tests to make sure we can evaluate an energy value given a set of Legendre series basis set coefficients. The energy value should be the same now matter how many times we take it, and should change if we change any of the basis set coefficients.'''
        test_axis = [i/200.0 -1 for i in range(401)]
        grid = pydinger.Grid(test_axis, False)
        grid.set_N(50)
        grid.set_v(1.0)
        grid.set_c(1.0)
        grid.get_hmat()
        energy = grid.get_energy()
        e2 = grid.get_energy()
        assert len(grid.coefficients) == grid.N
        for item in grid.coefficients:
            assert item - 1.0 < 0.00001
        assert isinstance(energy, float)
        assert energy - e2 < 0.00001

    def test_get_additions_fourier(self):
        '''This tests one of two main steps in the variational method I employ, for the Fourier basis set. I add 5% of each coefficient's value to it, then check the new energy from that change alone. If the change decreases the energy, we keep it. This change is tracked in a separate array (of zeros, 1s and -1s) the same length as the coefficient vector.'''
        test_axis = [i/200.0 -1 for i in range(401)]
        grid = pydinger.Grid(test_axis, True)
        grid.set_N(50)
        grid.set_v(1.0)
        grid.set_c(1.0)
        grid.coefficients = np.ones(grid.N)
        grid.get_hmat()
        grid.get_energy()
        grid.get_additions(True)
        original_coeffs = copy.deepcopy(grid.coefficients)
        changed = False
        #there will almost certainly be changes
        for i in range(grid.N):
            if(grid.changes[i] != 0):
                changed = True
        assert changed == True
        #the values of coefficients should stay the same
        for i in range(grid.N):
            assert abs(original_coeffs[i] - grid.coefficients[i]) < 0.00001

    def test_get_additions_legendre(self):
        '''This tests one of two main steps in the variational method I employ, for the Legendre basis set. I add 5% of each coefficient's value to it, then check the new energy from that change alone. If the change decreases the energy, we keep it. This change is tracked in a separate array (of zeros, 1s and -1s) the same length as the coefficient vector.'''
        test_axis = [i/200.0 -1 for i in range(401)]
        grid = pydinger.Grid(test_axis, False)
        grid.set_N(50)
        grid.set_v(1.0)
        grid.set_c(1.0)
        grid.get_hmat()
        grid.get_energy()
        grid.get_additions()
        original_coeffs = copy.deepcopy(grid.coefficients)
        changed = False
        #there will almost certainly be some changes
        for i in range(grid.N):
            if(grid.changes[i] != 0):
                changed = True            
        assert changed == True
        #the values of coefficients should stay the same
        for i in range(grid.N):
            assert abs(original_coeffs[i] - grid.coefficients[i]) < 0.00001

    def test_get_subtractions_fourier(self):
        '''This tests one of two main steps in the variational method I employ, for the Fourier basis set. I subtract 5% of each coefficient's value from it, then check the new energy from that change alone. If the change decreases the energy, we keep it. This change is tracked in a separate array (of zeros, 1s and -1s) the same length as the coefficient vector.'''
        test_axis = [i/200.0 -1 for i in range(401)]
        grid = pydinger.Grid(test_axis, True)
        grid.set_N(50)
        grid.set_v(1.0)
        grid.set_c(1.0)
        grid.get_hmat()
        grid.get_energy()#this is to get some coefficients...
        original_coeffs = copy.deepcopy(grid.coefficients)
        grid.get_subtractions(True)
        changed = False
        #there will certainly be some changes
        for i in range(grid.N):
            if(grid.changes[i] == -1):
                changed = True            
        assert changed == True
        #the values of coefficients should stay the same
        for i in range(grid.N):
            assert abs(original_coeffs[i] - grid.coefficients[i]) < 0.00001

    def test_get_subtractions_legendre(self):
        '''This tests one of two main steps in the variational method I employ, for the Legendre basis set. I subtract 5% of each coefficient's value from it, then check the new energy from that change alone. If the change decreases the energy, we keep it. This change is tracked in a separate array (of zeros, 1s and -1s) the same length as the coefficient vector.'''
        test_axis = [i/200.0 -1 for i in range(401)]
        grid = pydinger.Grid(test_axis, False)
        grid.set_N(50)
        grid.set_v(1.0)
        grid.set_c(1.0)
        grid.get_hmat()
        grid.get_energy()#this is to get some coefficients...
        original_coeffs = copy.deepcopy(grid.coefficients)
        grid.get_subtractions(True)
        changed = False
        #there will certainly be some changes
        for i in range(grid.N):
            if(grid.changes[i] == -1):
                changed = True            
        assert changed == True
        #the values of coefficients should stay the same
        for i in range(grid.N):
            assert abs(original_coeffs[i] - grid.coefficients[i]) < 0.00001

    def test_get_both(self):
        '''This is a sanity check to make sure that I'm not scheduling and addition AND subtraction for any given coefficient.'''
        test_axis = [i/200.0 -1 for i in range(401)]
        grid = pydinger.Grid(test_axis, False)
        grid.set_N(50)
        grid.set_v(1.0)
        grid.set_c(1.0)
        grid.get_hmat()
        grid.get_energy()#this is to get some coefficients...
        original_coeffs = copy.deepcopy(grid.coefficients)
        grid.get_subtractions()
        sub_array = copy.deepcopy(grid.changes)
        grid.get_additions()
        add_array = grid.changes
        for i in range(grid.N):
            if(sub_array[i] == -1):
                #make sure that we keep all the subtractions we noted
                assert(add_array[i] == -1)
            if(sub_array[i] == 0):
                #make sure that we either kept a 0 or added a 1
                assert(add_array[i] == 0 or add_array[i] == 1)
                    
        
