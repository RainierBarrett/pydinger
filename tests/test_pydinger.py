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
        assert potentials.dim == 1
        assert len(potentials.axes[0]) == 200
        assert potentials.axes[0][0] - 0.000100 < 0.000001
        assert potentials.axes[0][199] - 1.990000 < 0.000001
        print(potentials.values[0])

    def test_read_table_2D(self):
        '''This tests to make sure we can read in a 2D potential file correctly.'''
        #Do I even need this???
        testfile = '2D_test.txt'
        potentials = pydinger.read_file(testfile, 2)
        assert potentials.dim == 2
        assert len(potentials.axes[0]) == 121
        assert potentials.axes[0][0] - 0.000100 < 0.000001
        assert potentials.axes[0][120] - 2.0001 < 0.000001
        assert potentials.axes[1][0] - 0.000100 < 0.000001
        assert potentials.axes[1][120] - 2.0001 < 0.000001
        assert len(potentials.values) == 121
        assert potentials.values[0] - 0.00014142135623730951 < 0.000001
        assert potentials.values[60] - 1.4143549837293323 <  0.000001
        assert potentials.values[120] - 2.8285685461024275 <  0.000001

    def test_read_input(self):
        '''This tests to make sure we can read an input file and fetch the appropriate potential file to go with it.'''
        testfile = 'fourier_test_input.txt'
        potentials = pydinger.read_input(testfile)
        #check that we read the potentials OK
        assert potentials.dim == 1
        assert potentials.dim == 1
        assert len(potentials.axes[0]) == 200
        assert potentials.axes[0][0] - 0.000100 < 0.000001
        assert potentials.axes[0][199] - 1.990000 < 0.000001
        print(potentials.values[0])
        #check that we have the right values for our constants
        assert potentials.c - 1.0 < 0.000001
        assert potentials.N == 100
        assert potentials.fourier == True

        testfile = 'lagrange_test_input.txt'
        potentials = pydinger.read_input(testfile)
        assert potentials.dim == 1
        assert potentials.dim == 1
        assert len(potentials.axes[0]) == 200
        assert potentials.axes[0][0] - 0.000100 < 0.000001
        assert potentials.axes[0][199] - 1.990000 < 0.000001
        print(potentials.values[0])
        #check that we have the right values for our constants
        assert potentials.c - 1.5 < 0.000001
        assert potentials.N == 10
        assert potentials.fourier == False
        
