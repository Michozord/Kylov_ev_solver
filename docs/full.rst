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

		``**kwargs``
			kwargs for generation of ``ngsolve.H1`` solution space.


``solve(self)``
^^^^^^^^^^^^^^^^^^^^^^^^^
	Core method, that performs the Krylov iteration to compute eigenvalues :math:`\omega^2`
	with corresponding eigenvectors. It stores the results of steps between ``m_min`` and ``m_max``
	in the ``KrylovSolver.results`` property.


``compute_true_eigvals(self)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	Computes true eigenvalues of the :math:`M^{-1} S` matrix. If this method has been called, 
	true eigenvalues are added to plots in ``KrylovSolver.plot_results()`` method.
	
	**NOTE**: This method should be used in small-scale examples for comparison of the
	results of the Krylov iteration with true eigenvalues only! In large-scale 
	examples it ruins the performance of the whole method, since it uses direct 
	solver on large matrices :math:`S` and :math:`M`.
	
	
``plot(self, start: float, end: float, title: str="", plot_filter: bool=True, label_om: str=r"$\omega$", label_step: str=r"$k$", label_filter: str=r"$|\tilde{\beta}_{\vec{\alpha}}(\omega)|$", ev_marker: str="x", ev_color: str="blue", filter_plot_kwargs: dict={"color":"red"})``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	Generates plot presenting obtained resonances (:math:`\omega`). The resonance-axis
	is scaled in :math:`\omega` (presents square roots of eigenvalues).

	**Parameters:**
	
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
			Dictionary with kwargs for the plot of the filter function. All kwargs of the ``plt.plot()`` method are supported. The default is ``{"color":"red"}``.

	**Raises:**
		``RuntimeError``
			If there are no results in the ``KrylovSolver``. Use ``solve()`` method and try again.


``plot2(self, start: float, end: float, title: str="", plot_filter: bool=True, label_om: str=r"$\omega^2$", label_step: str=r"$k$", label_filter: str=r"$|\tilde{\beta}_{\vec{\alpha}}(\omega^2)|$", ev_marker: str="x", ev_color: str="blue", filter_plot_kwargs: dict={"color":"red"})``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	Generates plot presenting obtained eigenvalues (:math:`\omega^2`). The eigenvalue-axis is scaled in :math:`\omega^2` (presents eigenvalues).

    **Parameters**
        see method ``KrylovSolver.plot()``

    **Raises**
        see method ``KrylovSolver.plot()``
		
		
``get_single_result(self, ev: float, k: int=-1) -> tuple[float, np.array]``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	Returns computed eigenvalue closest to given ev with its eigenvector 
	after `k`-th step of the Krylov iteration.

    **Parameters:**

        ``ev : float``
            Eigenvalue (:math:`\omega^2`), to which closest value should be returned.
        ``k : int, optional``
            Step of the itereation. Use -1 for last iteration. The default is -1.

    **Raises:**
        ``RuntimeError``
            If there are no results in the KrylovSolver. Use solve() method and try again.
        ``ValueError``
            If given step k is not in stored results.

    **Returns:**
        ``float``
            Eigenvalue (:math:`\omega^2`) in results of the k-th step closest to ev.
        ``np.array``
            Eigenvector to the sought eigenvalue.


``Results``
---------------------
A simple dictionary-like class to store results of the Krylov iteration. 

- Key ``k`` is the number of iteration between ``m_min`` and ``m_max`` (-1 refers to the last iteration).
- Value is a ``Tuple[np.ndarray, np.ndarray]]``. The first array (``eigvals``) contains obtained eigenvalues (:math:`\omega^2` in this step). The second one (``eigvecs``) contains eigenvectors in columns. ``eigvecs[:,i]`` is an eigenvector to ``eigvals[i]``.



``FilterGenerator``
------------------------
This dataclass contains methods, that generate weights (:math:`\alpha`) in standard way: by :math:`L_2` minimization or collocation / least-squares in Chebyshev nodes.

**Parameters:**
	``_L: int``
		number of time-steps
	``_tau: float``
		time-step
	``_om_min: float``
		start of the target interval
	``_om_max: float``
		end of the target interval
	``_om_end: float``
		end of the controlled interval

``chebyshev(self, K: int) -> Filter``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	Returns weights (as a ``Filter``) obtained by the collocation or least-squares 
	fitting in Chebyshev nodes in :math:`\omega^2`.

	**Parameters:**
	
        ``K : int``
            Number of nodes.

    **Returns:**
        ``Filter``
			A ``Filter`` object with computed weights.


``l2(self, K: Optional[int] = 20) -> Filter``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	Returns weights (as a ``Filter``) obtained by :math:`L_2` minimization.

    **Parameters:**
        ``K : Optional[int], optional``
            Number of sample points for numerical quadrature in each unit. The default is 20.

    **Returns:**
        ``Filter``
			A ``Filter`` object with computed weights.

``fourier(self) -> Filter``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   Returns weights (as Filter) obtained by inverse Fourier transform. **Note**: this method works for negative Laplacian problem only! 

    **Returns:**
        ``Filter``
			A ``Filter`` object with computed weights.


``plot_chebyshev_nodes(self, N: int, ax: Optional[Axes] = None, marker="x", **kwargs) -> Axes``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	Plots ``N`` Chebyshev nodes in :math:`\omega^2` on :math:`omega`-scaled axis.

    **Parameters:**
        ``N : int``
            Number of nodes.
        ``ax : Optional[Axes], optional``
            An ``Axes`` object, where nodes should be plotted. If ``None``, plot is on 
            a new axis. The default is ``None``.
        ``marker : str, optional``
            A ``matplotlib`` marker. The default is ``"x"``.
        ``**kwargs``
            kwargs for ``matplotlib.axes.Axes.plot()`` method.

    **Returns:**
        ``Axes``
			``Axes`` object with plotted nodes.



``Filter``
------------------
Class to store filter as a numpy ``ndarray`` (actually evaluation of weights :math:`\alpha` at points :math:`0, \tau, 2\tau, ..., \tau (L-1)` with its parameters: time-step ``tau``, ``omega_end``, number of time-steps ``L`` and derivation method of the filter (``FilterType``).

``__new__(cls, array_input, filter_type, om_end: float, tau: float)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Constructor of a new filter.

    **Parameters:**
        ``array_input``
            Evaluation of weights alpha at points :math:`0, \tau, 2\tau, ..., \tau (L-1)` as a ``list``, ``tuple`` or anything that can be casted to a numpy ``ndarray``.
        ``filter_type : FilterType``
			Filter generation method.
        ``om_end : float``
            :math:`\omega_{\mathrm{end}}`, ``om_end > 0``.
        ``tau : float``
            Time-step, ``tau > 0``.

    **Returns:**
        ``obj``
			A new ``Filter`` object.


``plot(self, start: Optional[float] = 0, end: Optional[float] = None, ax: Optional[Axes] = None, num: Optional[int] = 10000, **kwargs) -> Axes``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This method plots filter function :math:`\beta(\omega)`. The method creates new axis or creates plot on given one.

    **Parameters:**
        ``start : Optional[float], optional``
            Start of the plot. The default is 0.
        ``end : Optional[float], optional``
            End of the plot. The default is ``None``: in this case ``end = om_end``.
        ``ax : Optional[Axes], optional``
            An ``Axes`` object, where the plot is created, if not None. Otherwise the method 
            plots on a new axis. The default is ``None``.
        ``num : Optional[int], optional``
            Fineness of the plot, i.e., number of sample points in the interval 
            (``start``, ``end``). The default is 10000.
        ``**kwargs``
            Kwargs for ``matplotlib.axes.Axes.plot()`` method.

    **Returns:**
        ``Axes``
            ``Axes`` object with the plot.



``plot2(self, start: Optional[int] = 0, end: Optional[int] = None, ax: Optional[Axes] = None, num: Optional[int] = 10000, **kwargs) -> Axes:``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Copy of the function ``Filter.plot()``. The only exception is that it plots :math:`\beta(\omega^2)`, not :math:`\beta(\omega)`.

    **Parameters:**
        see ``Filter.plot()``.
        

    **Returns:**
        see ``Filter.plot()``.


``FilterType``
-------------------
A simple ``Enum`` to distinguish types of filter functions.