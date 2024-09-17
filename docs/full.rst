Classes and methods 
=======================

Class ``KrylovSolver``
------------------------
This class performs FEM with Krylov iteration: discretizes the solution space, computes the discretization matrices and solves for its eigenpairs using Krylov iteration.
    
**Initialization parameters**:

..  code-block::
	s: Callable[[ngsolve.comp.ProxyFunction, ngsolve.comp.ProxyFunction], ngsolve.fem.CoefficientFunction]
	
Left-hand-site of the weak formulation of the problem to solve.

..  code-block::
	m : Callable[[ngsolve.comp.ProxyFunction, ngsolve.comp.ProxyFunction], ngsolve.fem.CoefficientFunction]

Right-hand-site of the weak formulation of the problem to solve.

..  code-block::
	mesh : ngsolve.comp.Mesh

A ``Mesh`` object of the discretized domain.

..  code-block::
	L : int

``L > 0``. Number of time-steps in each iteation.
..  code-block::
	tau : tau

``tau > 0``. Size of each time-step.

..  code-block::
	alpha : Filter

A ``Filter`` object representing discrete filter function (dff).

..  code-block::
	m_min : int, optional

From this iteration onwards the results are saved in ``KrylovSolver.results``. The default is 2.

..  code-block::
	m_max : int, optional

Maximal number of Krylov iterations. The default is 30.

**Methods:**
..  code-block::
	discretize(self, **kwargs)

This method discretizes the problem: creates solution space with its basis, prepares matrices M and S.

Parameters
----------

..  code-block::
	**kwargs 

kwargs for generation of ``ngsolve.H1`` solution space.




Class ``Results``
--------------------
A simple dictionary-like class to store results of the Krylov iteration. 

- Key `k` is the number of iteration between `m_min` and `m_max` (-1 refers to the last iteration).
- Value is a `Tuple[np.ndarray, np.ndarray]]`. The first array (`eigvals`) contains obtained eigenvalues (:math:`\omega^2` in this step). The second one (`eigvecs`) contains eigenvectors in columns. eigvecs[:,i] is an eigenvector to eigvals[i].
