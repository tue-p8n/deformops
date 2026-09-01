"""Machinery every operator and backend shares; not public API.

It holds the JIT build, launch heuristics, operand validation and shared CUDA
headers, so no operator imports another to get them.
"""
