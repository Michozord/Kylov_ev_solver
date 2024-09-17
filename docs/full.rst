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


def plot(self, start: float, end: float, title: str="", plot_filter: bool=True, 
				 label_om: str=r"$\omega$", label_step: str=r"$k$", 
				 label_filter: str=r"$|\tilde{\beta}_{\vec{\alpha}}(\omega)|$",
				 ev_marker: str="x", ev_color: str="blue",
				 filter_plot_kwargs: dict={"color":"red"}):
	"""
	Generates plot presenting obtained resonances (omegas). The resonance-axis
	is scaled in omega (presents square roots of eigenvalues).

	Parameters
	----------
	start : float
		start point of the plot.
	end : float
		end point of the plot.
	title : str, optional
		Title of the plot. The default is "".
	plot_filter : bool, optional
		If True, plot of the filter function is appended to the plot. The default is True.
	label_om : str, optional
		Label on the omega (horizontal)-axis. The default is r"$\\omega$".
	label_step : str, optional
		Label on the iteration-step k (vertical left)-axis. The default is r"$k$".
	label_filter : str, optional
		Label on the filter function (vertical right)-axis. The default is r"$|\\tilde{\\beta}_{\\vec{\\alpha}}(\\omega)|$".
	ev_marker : str, optional
		Marker for omegas. The default is "x".
	ev_color : str, optional
		Color of eigenvalues. The default is "blue".
	filter_plot_kwargs : dict, optional
		Dictionary with kwargs for the plot of the filter function. All kwargs of plt.plot method are supported. The default is {"color":"red"}.

	Raises
	------
	RuntimeError
		If there are no results in the KrylovSolver. Use solve() method and try again.

	"""

Class ``Results``
--------------------
A simple dictionary-like class to store results of the Krylov iteration. 

- Key `k` is the number of iteration between `m_min` and `m_max` (-1 refers to the last iteration).
- Value is a `Tuple[np.ndarray, np.ndarray]]`. The first array (`eigvals`) contains obtained eigenvalues (:math:`\omega^2` in this step). The second one (`eigvecs`) contains eigenvectors in columns. eigvecs[:,i] is an eigenvector to eigvals[i].
