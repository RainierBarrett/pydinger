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
        potentials = pydinger.read_file(testfile)
        assert len(potentials.axis) == 200
        assert potentials.axis[0] - 0.000100 < 0.000001
        assert potentials.axis[199] - 1.990000 < 0.000001
        assert potentials.period - 2.0  < 0.000001
        assert potentials.offset - 0.000100  < 0.000001

    def test_read_input(self):
        '''This tests to make sure we can read an input file and fetch the appropriate potential file to go with it.'''
        testfile = 'fourier_test_input.txt'
        potentials = pydinger.read_input(testfile)
        #check that we read the potentials OK
        assert len(potentials.axis) == 200
        assert potentials.axis[0] - 0.000100 < 0.000001
        assert potentials.axis[199] - 1.990000 < 0.000001
        #print(potentials.values[0])
        #check that we have the right values for our constants
        assert potentials.c - 1.0 < 0.000001
        assert potentials.N == 100
        assert potentials.v - 2.0 < 0.000001
        assert potentials.fourier == True
        for i in range(len(potentials.axis)):
            assert (potentials.wavefunc[i] - (potentials.axis[i]**4 - potentials.axis[i]**2) < 0.000001)
        testfile = 'legendre_test_input.txt'
        potentials = pydinger.read_input(testfile)
        assert len(potentials.axis) == 200
        assert potentials.axis[0] - 0.000100 < 0.000001
        assert potentials.axis[199] - 1.990000 < 0.000001
        #check that we have the right values for our constants
        assert potentials.c - 1.5 < 0.000001
        assert potentials.N == 10
        assert potentials.fourier == False
        for i in range(len(potentials.axis)):
            assert (potentials.wavefunc[i] - (potentials.axis[i]**4 - potentials.axis[i]**2) < 0.000001)

            
    def test_get_coeffs_fourier(self):
        '''This tests that we are able to recover a well-behaved function's Fourier polynomial coefficients. Seeing whether they're 'good' is beyond the scope of unit testing, but I made sure to play around with them in a Jupyter Notebook, and it works well.'''
        testfile = 'fourier_test_input.txt'
        potentials = pydinger.read_input(testfile)
        #this is a nice-looking, well-behaved (easy to converge upon) function for a fourier series
        test_wavefunc = potentials.axis*2 + potentials.axis**2 - potentials.axis**3
        potentials.get_coefficients(test_wavefunc)#should calculate the fourier coefficients
        assert len(potentials.coefficients) == potentials.N
        values = potentials.get_values(test_wavefunc)
        diffsquare = 0
        for i in range(len(values)):
            diffsquare +=(values[i] - test_wavefunc[i])**2
        assert(diffsquare < 0.05)#that should be pretty accurate

    def test_get_coeffs_legendre(self):
        '''This will test if we are able to get the same function as used in the Fourier test by using Legendre polynomials instead, and properly plot from said coefficients. Like the Fourier test, how good this fits is a bit outside the scope of unit testing, but I've looked at the plots here in a Jupyter Notebook and it also works well.'''
        testfile = 'legendre_test_input.txt'
        potentials = pydinger.read_input(testfile)
        test_wavefunc = potentials.axis*2 + potentials.axis**2 - potentials.axis**3
        potentials.get_coefficients(test_wavefunc)
        assert len(potentials.coefficients) == potentials.N
        values = potentials.get_values( test_wavefunc )
        assert len(values) == len(potentials.axis)
        diffsquare = 0
        for i in range(len(values)):
            diffsquare +=(values[i] - test_wavefunc[i])**2
        assert(diffsquare < 0.05)#that should be pretty accurate


    def test_get_hmat_fourier(self):
        '''This tests to make sure we can obtain the Hamiltonian matrix from our Fourier basis representation of a wavefunction. Specifically, that the matrix is diagonal, which occurs naturally due to the orthonormality of this basis set.'''
        testfile = 'fourier_test_input.txt'
        potentials = pydinger.read_input(testfile)
        test_wavefunc = np.sin(potentials.axis)
        potentials.get_coefficients(test_wavefunc)#now we have the original coefficients
        potentials.get_hmat()#this creates the hamiltonian matrix
        assert len(potentials.hmat) == potentials.N
        assert len(potentials.hmat[0]) == potentials.N
        #the matrix will be diagonal, but the (0,0) entry will be 0 here, so:
        for i in range(1,potentials.N):
            assert potentials.hmat[i][i] != 0

        
    def test_apply_H_fourier(self):
        '''This will test to see if we can apply the Hamiltonian (in matrix representation) to a wavefunction with the Fourier series basis set.'''
        testfile = 'fourier_test_input.txt'
        potentials = pydinger.read_input(testfile)
        test_wavefunc = np.sin(potentials.axis)
        potentials.get_coefficients(test_wavefunc)
        potentials.get_hmat()
        old_coeffs = copy.deepcopy(potentials.coefficients)
        potentials.apply_H()#now we should have operated the Hamiltonian
        for i in range(potentials.N):#these should all change now, with our wavefunction
            assert potentials.coefficients[i] != old_coeffs[i]

    def test_apply_H_legendre(self):
        '''This tests that we can apply the Hamiltonian to a wavefunction with the Legendre polynomial basis set.'''
        testfile = 'legendre_test_input.txt'
        potentials = pydinger.read_input(testfile)
        x = potentials.axis - 1 #because the input file was from 0 to 2, this will be from -1 to 1
        test_wavefunc = x**2 -2*x**3 + 4
        potentials.get_coefficients(test_wavefunc)
        #since the legendre polynomial module in numpy has a built-in method to just GET the second derivative basis set coefficients, I just use that.
        #note that we have to pad the new list it makes, because two of the coefficients will become zero when taking the second derivative, naturally...
        old_coeffs = copy.deepcopy(potentials.coefficients)
        potentials.apply_H()
        for i in range(potentials.N):#these should all change now, with our wavefunction
            assert potentials.coefficients[i] != old_coeffs[i]


            
        #ASIDE: I'm really not sure how else to test these bad boys and keep generality. 
