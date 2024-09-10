.. krylovevsolver documentation master file, created by
   sphinx-quickstart on Fri Sep  6 14:58:54 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

1. Welcome to Krylov eigenvalue solver (``krylovevsolver``) documentation
=========================================================================

This Python package is designed for numerically solving eigenvalue problems of linear partial differential operators, such as the Laplacian operator. It utilizes the finite element method, specifically employing the ``netgen/ngsolve`` packages (see here_), to define domains, boundaries, and discretize the problem.

.. _here: https://ngsolve.org

By applying special filter functions and Krylov iteration to the generated discretization matrices, the package computes the eigenvalues of the operator within a specified region of interest (an interval set by the user), along with the corresponding eigenvectors (eigenfunctions). This approach significantly reduces the size of the matrix eigenvalue problem after discretization, thereby lowering computational costs. Krylov iteration is particularly suitable for large-scale problems with a high number of degrees of freedom.


2. Mathematical background
==========================

sfgdsfgdsfg

3. Installation 
===============

sfgadfgsdgfdsfg



4. Solving eigenvalue problems
==============================
dfgsdfgdsfgdsfg

.. toctree::
   :maxdepth: 2
   :caption: Contents:

