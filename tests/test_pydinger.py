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
