.. krylovevsolver documentation master file, created by
   sphinx-quickstart on Fri Sep  6 14:58:54 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

``krylovevsolver``. A Krylov eigenvalue solver in Python
=========================================================

This Python package is designed for numerically solving eigenvalue problems of linear partial differential operators, such as the Laplacian operator. It utilizes the finite element method, specifically employing the ``netgen`` and ``ngsolve`` packages (see here), to define domains, boundaries, and discretize the problem.

.. _here: https://ngsolve.org

By applying special filter functions and Krylov iteration to the generated discretization matrices, the package computes the eigenvalues of the operator within a specified region of interest (an interval set by the user), along with the corresponding eigenvectors (eigenfunctions). This approach significantly reduces the size of the matrix eigenvalue problem after discretization, thereby lowering computational costs. Krylov iteration is particularly suitable for large-scale problems with a high number of degrees of freedom.

.. image:: images/wave.png
   :width: 600
Visualisation of an eigenfunction of the negative Laplacian on a simple rectangular domain.


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   background.rst
   installation.rst
   full.rst

