r"""Alternative CUDA MSDA kernel configurations for comparison.

Each compiled module differs from production in gradient accumulation or
softmax reduction; ``msda_morton`` and ``msda_hilbert`` reorder queries around
the production kernel. :class:`deformops.registry.Backend` cannot select them:
pass a module's ``Backend()`` as the ``backend`` of
:class:`deformops.nn.MultiScaleDeformAttn2d` or call its ``forward``.
"""
