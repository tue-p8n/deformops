r"""Drop-in integrations with third-party libraries.

Each module adapts one library. Nothing is re-exported, so importing the package
imports no third-party library; import a module directly::

    from deformops.integrations.torchvision import deform_conv2d
    from deformops.integrations.transformers import MultiScaleDeformableAttention
    from deformops.integrations.mmcv import MultiScaleDeformableAttention
"""
