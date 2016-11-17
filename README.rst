===============================
pydinger
===============================

.. image:: https://img.shields.io/travis/RainierBarrett/pydinger.svg
           :target: https://travis-ci.org/RainierBarrett/pydinger

.. image:: https://coveralls.io/repos/github/RainierBarrett/pydinger/badge.png?branch=master
	   :target: https://coveralls.io/github/RainierBarrett/pydinger?branch=master
		    

This is a Python program to solve Schroedinger's equation (in 1D) for the ground-state wavefunction. It allows for the use of either Fourier series, or Legendre polynomial basis sets, and applies the variational principle to minimize energy. When fitting to an input wavefuntion, it uses Riemann sum integration for taking the integral needed to find coefficient values for the Fourier series, and a builtin function to deal with Legendre coefficients.

For the Fourier basis set, I simply solved for the Hamiltonian matrix entries by hand, and filled a matrix with the values needed to apply the Laplacian. Each time I "apply" the Hamiltonian, I actually just multiply by -c times the (i,i)th entry in the matrix, and add V times the current coefficient. This has the same effect as taking the inner product, but is much faster than matrix multiplication. I save time in a similar fashion with the Legendre polynomial basis set. Instead of taking inner products directly, I apply the Laplacian by using a builtin second-derivative function in the numpy.polynommial.legendre package, and again just add V times the original coefficient to each value.

These tricks work because both Fourier series and Legendre polynomials are mutually orthogonal on their domains, which results in a "naturally" diagonal Hamiltonian matrix, so matrix multiplication of H with the coefficient vector is the same as just scaling each ith coefficient by the (i,i)th entry in the matrix.


Running Unit Tests
--------
To run the unit tests for this program, make sure you have a suitable tox environment, then simply invoke the command "tox" from within the source directory. To test for coverage, run the command "coverage run --source=pydinger/pydinger.py setup.py test", and to check the coverage, run "coverage report -m".

Running the Program
--------
To run this program, simply change directories into the internal "pydinger" directory and run the command "python main.py". This will run a 10,000-step cutoff calculation, and report the coefficients of the resultant wavefuntion. The cutoff value is to ensure that the variational method I employ (just a gradient descent on each coefficient value) won't get stuck in an endless loop. It seems to work and terminates before the cutoff when I run it, so this is mostly just a precaution.

NOTE: Make sure that you edit the 'input.txt' file included within the internal pydinger folder (the one with main.py in it), and ensure that it is pointing at a valid path to your axis file. For examples of both of these files, see fourier_test_input.txt, and 1D_test.txt, respectively. Note that in the input file, "BASIS 1" corresponds to using the Fourier basis set, and "BASIS 0" corresponds to using Legendre. If no basis is specified, Fourier will be used by default. Be sure to format your input files in the same way, including ordering, as the example input files!

PS: I seem to get a reasonable answer when I use the Fourier basis set on my input.txt, but when I use the Legendre basis set, I get a wildly different, massively negative energy value. I am still unsure as to what has caused this issue.

TODO
--------
* Fix energy issue with Legendre polynomials.
* Possibly add function to produce graphs of the function as it is being varied?

License
---------
.. image:: https://www.gnu.org/graphics/gplv3-127x51.png

This program is Free Software: You can use, study share and improve it at your
will. Specifically you can redistribute and/or modify it under the terms of the
[GNU General Public License](https://www.gnu.org/licenses/gpl.html) as
published by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Credits
---------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

