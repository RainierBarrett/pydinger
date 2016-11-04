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

    def test_read_energy_1D(self):
        '''this tests to make sure we can read in a 1D potential file'''
        testfile = '1D_test.txt'
        potentials = pydinger.read_file(testfile, 1)
        assert potentials.dim == 1
        assert len(potentials.axes[0]) == 200
        assert potentials.axes[0][0] - 0.000100 < 0.000001
        assert potentials.axes[0][199] - 1.990000 < 0.000001
        print(potentials.values[0])
