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
        potentials = pydinger.read_file(testfile, 1)
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
        print(potentials.values[0])
        #check that we have the right values for our constants
        assert potentials.c - 1.0 < 0.000001
        assert potentials.N == 100
        assert potentials.fourier == True

        testfile = 'lagrange_test_input.txt'
        potentials = pydinger.read_input(testfile)
        assert len(potentials.axis) == 200
        assert potentials.axis[0] - 0.000100 < 0.000001
        assert potentials.axis[199] - 1.990000 < 0.000001
        print(potentials.values)
        #check that we have the right values for our constants
        assert potentials.c - 1.5 < 0.000001
        assert potentials.N == 10
        assert potentials.fourier == False
        
    def test_get_coeffs_fourier(self):
        '''This tests that we are able to recover a well-behaved function's Fourier polynomial coefficients. Seeing whether they're 'good' is beyond the scope of unit testing, but I made sure to play around with them in a Jupyter Notebook, and it works well.'''
        testfile = 'fourier_test_input.txt'
        potentials = pydinger.read_input(testfile)
        #this is a nice-looking, well-behaved (easy to converge upon) function for a fourier series
        test_wavefunc = potentials.axis*2 + potentials.axis**2 - potentials.axis**3
        print("TEST_WAVEFUNCTION IS {}".format(test_wavefunc))
        potentials.get_coefficients(test_wavefunc)#should calculate the fourier coefficients
        assert len(potentials.coefficients) == potentials.N
        values = potentials.get_fourier_values(test_wavefunc)
        print("VALUES IS {}".format(values))
        diffsquare = 0
        #print(potentials.axis[0])
        for i in range(len(values)):
            diffsquare +=(values[i] - test_wavefunc[i])**2
            print("values[{}] is {}".format(i, values[i]))
            print("and test_wavefunc[{}] is {}".format(i,test_wavefunc[i]))
#        assert(diffsquare < 10.0)

        
