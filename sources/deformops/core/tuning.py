"""Static launch layout for the CUDA multi-scale deformable attention kernels.

Picks ``(channels_per_thread, threads_per_block)`` from the shapes, so the kernels
need no runtime autotuning. The block caps are targets a shape may exceed
(``G = 8``, ``D = 128`` gives 512 in the backward), and the kernels' launch bounds
are sized for that. See ``docs/kernels.md``.
"""

import functools

__all__ = [
    "find_factors",
    "backward_thread_layout",
    "forward_thread_layout",
]


def find_factors(n: int) -> list[int]:
    r"""Find all integer factors of a positive number.

    Parameters
    ----------
    n
        The positive integer to factorize.

    Returns
    -------
    A sorted list of all unique integer divisors of ``n``.

    Raises
    ------
    ValueError
        If ``n`` is negative.
    """
    if n < 0:
        msg = f"Expected a positive integer. Got: {n}!"
        raise ValueError(msg)
    result: set[int] = set()
    for i in range(1, int(n**0.5) + 1):
        div, mod = divmod(n, i)
        if mod == 0:
            result |= {i, div}
    return sorted(result)


def _concrete(value: int) -> int:
    """Force a symbolic size to ``int``.

    The memoized layouts cannot hash a ``SymInt``, and a launch needs a concrete
    layout anyway, so the traced graph specializes on this size.
    """
    return int(value)


def forward_thread_layout(B: int, Q: int, G: int, D: int) -> tuple[int, int]:
    """Resolve symbolic sizes, then pick via :func:`_forward_thread_layout`."""
    return _forward_thread_layout(
        _concrete(B), _concrete(Q), _concrete(G), _concrete(D)
    )


@functools.lru_cache(maxsize=128)
def _forward_thread_layout(B: int, Q: int, G: int, D: int) -> tuple[int, int]:
    r"""Pick ``(channels_per_thread, threads_per_block)`` for the forward kernel.

    Parameters
    ----------
    B
        Batch size per kernel launch, at most ``chunk_step``.
    Q
        Queries per batch.
    G
        Number of attention heads.
    D
        Number of channels per head.

    Returns
    -------
    A tuple of ``(channels_per_thread, threads_per_block)``.

    Notes
    -----
    * ``channels_per_thread = 8``, for vectorized loads across 8 channels.
    * ``threads_per_block <= 512``, from the largest ``queries_per_block <= 64``.
    """
    channels_per_thread = 8
    queries_per_block = 1
    for m in find_factors(B * Q):
        if m <= 64 and (m * G * D // channels_per_thread) <= 512:  # noqa: PLR2004
            queries_per_block = m
    threads_per_block = queries_per_block * G * D // channels_per_thread
    return channels_per_thread, threads_per_block


def backward_thread_layout(B: int, Q: int, G: int, D: int) -> tuple[int, int]:
    """Resolve symbolic sizes, then pick via :func:`_backward_thread_layout`."""
    return _backward_thread_layout(
        _concrete(B), _concrete(Q), _concrete(G), _concrete(D)
    )


@functools.lru_cache(maxsize=128)
def _backward_thread_layout(B: int, Q: int, G: int, D: int) -> tuple[int, int]:
    r"""Pick ``(channels_per_thread, threads_per_block)`` for the backward kernel.

    Parameters
    ----------
    B
        Batch size per kernel launch, at most ``chunk_step``.
    Q
        Queries per batch.
    G
        Number of attention heads.
    D
        Number of channels per head.

    Returns
    -------
    A tuple of ``(channels_per_thread, threads_per_block)``.

    Notes
    -----
    * ``channels_per_thread = 2 if D >= 64 else 1``: the backward accumulates three
      gradients at once, and a smaller slice avoids register spilling.
    * ``threads_per_block <= 256``, half the forward cap, for the same reason.
    """
    channels_per_thread = 2 if D >= 64 else 1  # noqa: PLR2004
    queries_per_block = 1
    for m in find_factors(B * Q):
        if m <= 64 and (m * G * D // channels_per_thread) <= 256:  # noqa: PLR2004
            queries_per_block = m
    threads_per_block = queries_per_block * G * D // channels_per_thread
    return channels_per_thread, threads_per_block
