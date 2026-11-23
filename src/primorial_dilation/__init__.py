# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright (c) 2026 Benjamin Edward Gray (https://doi.org/10.5281/zenodo.22899830)
# Required Notice: Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE.md)
# Commercial licensing inquiries: graybenjamin217@gmail.com

"""Reference implementation for the Primorial-Modulated Dilation Operator."""

from .operator import (
    PRIMES,
    active_sector_eigenvalues,
    build_T_herm,
    build_normalized_channel,
    build_prime_channel,
    compute_spectrum,
    crt_fiber_injection,
    euler_totient_primorial,
    exact_channel_inner_product,
    exact_nullity,
    exact_rank,
    exact_second_moment,
    exact_trace,
    g2_generators,
    g2_sector_projector,
    hilbert_schmidt_resolvent_norm_sq,
    primorial,
    stage_diagnostics,
    trace_moment_expansion,
)

__all__ = [
    "PRIMES",
    "primorial",
    "euler_totient_primorial",
    "exact_nullity",
    "exact_rank",
    "exact_trace",
    "exact_channel_inner_product",
    "exact_second_moment",
    "build_prime_channel",
    "build_normalized_channel",
    "build_T_herm",
    "g2_generators",
    "g2_sector_projector",
    "active_sector_eigenvalues",
    "stage_diagnostics",
    "compute_spectrum",
    "hilbert_schmidt_resolvent_norm_sq",
    "trace_moment_expansion",
    "crt_fiber_injection",
]
