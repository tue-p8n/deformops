def test_deform_conv2d_is_not_the_sampler() -> None:
    """Deformable convolution and sampling are distinct classes, not aliases."""
    from deformops.nn import DeformConv2d, DeformSample2d

    assert DeformConv2d is not DeformSample2d
    assert DeformConv2d.__module__.startswith("deformops.nn")
    assert DeformSample2d.__module__.startswith("deformops.nn")


def test_the_classic_operator_carries_a_convolution_weight() -> None:
    """The classic operator has a convolution weight; the sampler has none.

    The sampler's 3x3 weight belongs to its offset generator, so the check is on the
    ``(C_out, C_in / groups, kH, kW)`` shape, not on any 4-D parameter.
    """
    from deformops.nn import DeformConv2d, DeformSample2d

    layer = DeformConv2d(4, 8, 3, padding=1)
    assert layer.weight.shape == (8, 4, 3, 3)
    assert layer.bias is not None

    sampler = DeformSample2d(dim=64, kernel_size=3, groups=4)
    assert not hasattr(sampler, "weight")
