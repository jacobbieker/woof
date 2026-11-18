"""Gradients of a forecast's fit to observations, without an adjoint (EXPERIMENTAL).

Phase P0 of ``docs/dev/DIFFERENTIABLE-HYBRID-PLAN.md``: an observation cost
``J_o`` over a time window (:mod:`woof.grad.obs_cost`), the observation
readers it needs (:mod:`woof.grad.obs_stores`), and the ``woof grad``
command door (:mod:`woof.grad.cli`).

Everything here is opt-in tooling that reads finished forecasts. Nothing on
a forecast path imports this package, and no model word changes because it
exists: ``tests/test_grad_obs_cost.py`` checks the import graph.
"""
