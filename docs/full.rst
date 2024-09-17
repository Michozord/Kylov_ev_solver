Classes and methods
========================

``KrylovSolver`` 
----------------------
This class performs FEM with Krylov iteration: discretizes the solution space, computes the discretization matrices and solves for its eigenpairs using Krylov iteration.

``__init__(self, s: Callable[[ProxyFunction, ProxyFunction], CoefficientFunction], m: Callable[[ProxyFunction, ProxyFunction], CoefficientFunction], mesh: comp.Mesh, L: int, tau: float, alpha: Filter, m_min: int = 2, m_max: int = 30)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	**Parameters:**

		``s: Callable[[ngsolve.comp.ProxyFunction, ngsolve.comp.ProxyFunction], ngsolve.fem.CoefficientFunction]``
			Left-hand-site of the weak formulation of the problem to solve.
		``m : Callable[[ngsolve.comp.ProxyFunction, ngsolve.comp.ProxyFunction], ngsolve.fem.CoefficientFunction]``
			Right-hand-site of the weak formulation of the problem to solve.
		``mesh : ngsolve.comp.Mesh``
			A ``Mesh`` object of the discretized domain.
		``L : int``
			``L > 0``. Number of time-steps in each iteation.
		``tau : tau``
			``tau > 0``. Size of each time-step.
		``alpha : Filter``
			A ``Filter`` object representing discrete filter function (dff).
		``m_min : int, optional``
			From this iteration onwards the results are saved in ``KrylovSolver.results``. The default is 2.
		``m_max : int, optional``
			Maximal number of Krylov iterations. The default is 30.




``discretize(self, **kwargs)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	This method discretizes the problem: creates solution space with its basis, prepares matrices :math:`M` and :math:`S`.

	**Parameters:**

		``**kwargs ``
			kwargs for generation of ``ngsolve.H1`` solution space.


``solve(self)``
^^^^^^^^^^^^^^^^^^^^^^^^^
	Core method, that performs the Krylov iteration to compute eigenvalues :math:`\omega^2`
	with corresponding eigenvectors. It stores the results of steps between ``m_min`` and ``m_max``
	in the ``KrylovSolver.results`` property.


``compute_true_eigvals(self)``
^^^^^^^^^^^^^^^^^^
	Computes true eigenvalues of the :math:`M^-1 S` matrix. If this method has been called, 
	true eigenvalues are added to plots in ``KrylovSolver.plot_results()`` method.
	
	**NOTE**: This method should be used in small-scale examples for comparison of the
	results of the Krylov iteration with true eigenvalues only! In large-scale 
	examples it ruins the performance of the whole method, since it uses direct 
	solver on large matrices :math:`S` and :math:`M`.
	
	
``plot(self, start: float, end: float, title: str="", plot_filter: bool=True, 
                     label_om: str=r"$\omega$", label_step: str=r"$k$", 
                     label_filter: str=r"$|\tilde{\beta}_{\vec{\alpha}}(\omega)|$",
                     ev_marker: str="x", ev_color: str="blue",
                     filter_plot_kwargs: dict={"color":"red"})``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	Generates plot presenting obtained resonances (omegas). The resonance-axis
	is scaled in :math:`\omega` (presents square roots of eigenvalues).

	**Parameters**
	
		``start : float``
			start point of the plot.
		``end : float``
			end point of the plot.
		``title : str, optional``
			Title of the plot. The default is ``""``.
		``plot_filter : bool, optional``
			If ``True``, plot of the filter function is appended to the plot. The default is ``True``.
		``label_om : str, optional``
			Label on the omega (horizontal)-axis. The default is ``r"$\\omega$"``.
		``label_step : str, optional``
			Label on the iteration-step ``k`` (vertical left)-axis. The default is ``r"$k$"``.
		``label_filter : str, optional``
			Label on the filter function (vertical right)-axis. The default is ``r"$|\\tilde{\\beta}_{\\vec{\\alpha}}(\\omega)|$"``.
		``ev_marker : str, optional``
			Marker for omegas. The default is ``"x"``.
		``ev_color : str, optional``
			Color of eigenvalues. The default is ``"blue"``.
		``filter_plot_kwargs : dict, optional``
			Dictionary with kwargs for the plot of the filter function. All kwargs of plt.plot method are supported. The default is ``{"color":"red"}``.

	**Raises**
		``RuntimeError``
			If there are no results in the ``KrylovSolver``. Use ``solve()`` method and try again.


``plot2(self, start: float, end: float, title: str="", plot_filter: bool=True, 
                     label_om: str=r"$\omega^2$", label_step: str=r"$k$", 
                     label_filter: str=r"$|\tilde{\beta}_{\vec{\alpha}}(\omega^2)|$",
                     ev_marker: str="x", ev_color: str="blue",
                     filter_plot_kwargs: dict={"color":"red"})``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	Generates plot presenting obtained eigenvalues (omegas^2). The eigenvalue-axis is scaled in :math:`\omega^2` (presents eigenvalues).

    **Parameters**
        see method ``KrylovSolver.plot()``

    **Raises**
        see method ``KrylovSolver.plot()``

``Results``
---------------------
A simple dictionary-like class to store results of the Krylov iteration. 

- Key ``k`` is the number of iteration between ``m_min`` and ``m_max`` (-1 refers to the last iteration).
- Value is a ``Tuple[np.ndarray, np.ndarray]]``. The first array (``eigvals``) contains obtained eigenvalues (:math:`\omega^2` in this step). The second one (``eigvecs``) contains eigenvectors in columns. ``eigvecs[:,i]`` is an eigenvector to ``eigvals[i]``.
