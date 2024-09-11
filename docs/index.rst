.. krylovevsolver documentation master file, created by
   sphinx-quickstart on Fri Sep  6 14:58:54 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

1. Welcome to Krylov eigenvalue solver (``krylovevsolver``) documentation
=========================================================================

This Python package is designed for numerically solving eigenvalue problems of linear partial differential operators, such as the Laplacian operator. It utilizes the finite element method, specifically employing the ``netgen/ngsolve`` packages (see here_), to define domains, boundaries, and discretize the problem.

.. _here: https://ngsolve.org

By applying special filter functions and Krylov iteration to the generated discretization matrices, the package computes the eigenvalues of the operator within a specified region of interest (an interval set by the user), along with the corresponding eigenvectors (eigenfunctions). This approach significantly reduces the size of the matrix eigenvalue problem after discretization, thereby lowering computational costs. Krylov iteration is particularly suitable for large-scale problems with a high number of degrees of freedom.

.. image:: images/wave.png
   :width: 600
Visualisation of an eigenfunction of the negative Laplacian on a simple rectangular domain.


2. Mathematical background
==========================

Full description of the method including proofs and examples you can find in PDF here_.

.. _here: https://x.com

2.1. Finite element method
--------------------------

We seek eigenpairs :math:`(\omega^2, u)` (note that we use the notation :math:`\omega^2` for the eigenvalue) of a linear partial derivative operator :math:`L` on a domain :math:`\Omega` with given boundary conditions. This means we aim to find :math:`u` such that :math:`Lu = \omega^2 u`. We begin with the standard Finite Element Method (FEM) approach: defining a mesh, fixing a finite-dimensional solution space of hat functions (or optionally, splines) :math:`V_h`, and reformulating the problem in its weak form.

As an example, we consider the **negative Laplacian eigenvalue problem** with Neumann boundary conditions:

.. math::
	-\Delta u = \omega^2 u \quad \text{ in } \Omega,
.. math::
	\frac{\partial u}{\partial \nu} = 0 \quad\text{ on } \partial\Omega. 

Its weak form is:

.. math::
	\int_\Omega \nabla u \cdot \nabla \varphi \, dx = \omega_h^2 \int_\Omega u \varphi \, dx.
	
Using a fixed basis :math:`\varphi_1, \dots, \varphi_N` of the solutions space :math:`V_h`, we define the discretization matrices :math:`S` and :math:`M` as follows:

.. math::
	s_{ij} := \int_\Omega \nabla \varphi_i \cdot \nabla \varphi_j \, dx \quad \text{ and } \quad m_{ij} := \int_\Omega \varphi_i \varphi_j \, dx.
	
This leads to the discrete matrix eigenvalue problem:

.. math::
	Sv = \omega^2 Mv,

or, equivalently, 
	
.. math::
	M^{-1}Sv = \omega^2 v,
	
where :math:`v` denotes the coordinate vector of the eigenfunction.

2.2. Filtered time-domain solutions
-----------------------------------

Krylov eigenvalue solver finds eigenvalues within a specified **region of interest**, denoted as :math:`(\omega_\min^2, \omega_\max^2)`. We aim to construct a linear operator :math:`C` that shares the same eigenspaces as :math:`M^{-1}S`, but with different eigenvalues. Crucial to the contruction of the operator :math:`C` is the **weight vector** :math:`\Vec{\alpha}` and induced **discrete filter function** (**dff**) :math:`\beta(\omega)`. 

- If :math:`v` is an eigenvector of :math:`M^{-1}S` corresponding to an eigenvalue :math:`\omega^2 \in (\omega_\min^2, \omega_\max^2)`, then :math:`v` is an eigenvector of :math:`C` corresponding to a large eigenvalue :math:`\beta(\omega)`.
- If :math:`v` is an eigenvector of :math:`M^{-1}S` corresponding to an eigenvalue :math:`\omega^2 \notin (\omega_\min^2, \omega_\max^2)`, then :math:`v` is an eigenvector of :math:`C` corresponding to a close to zero eigenvalue :math:`\beta(\omega)`.

.. image:: images/dff.png
   :width: 600

For a detailed construction of this operator, see PDF_. It is impossible to push values of :math:`\beta` outside the region of interest close to 0, so we control the values of dff withinin a **control interval** denoted as :math:`(0, \omega_{\mathrm{end}}^2)` only. All eigenvalues of :math:`M^{-1}S` should lie in this interval: :math:`(0, \omega_{\mathrm{end}}^2)`. 
 
.. _PDF: https://x.com

In constructing :math:`C`, we perform simple time-stepping with :math:`L` time-steps of size :math:`\tau` to the end-time :math:`T = L\tau`. **CFL condition** requires, that :math:`\tau \leq 2/\omega_{\mathrm{end}}`. For computational efficiency, we recommend using :math:`\tau \lessapprox 2/\omega_{\mathrm{end}}`. A higher number of time-steps :math:`L` (or larger end-time :math:`L`) increases the reliability of the algorithm and improves the behavior of the dff, but linearly increases computation costs.


3. Installation 
===============

sfgadfgsdgfdsfg


4. Solving eigenvalue problems
==============================
dfgsdfgdsfgdsfg


.. toctree::
   :maxdepth: 2
   :caption: Contents:

