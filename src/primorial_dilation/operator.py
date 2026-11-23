# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright (c) 2026 Benjamin Edward Gray (https://doi.org/10.5281/zenodo.22899830)
# Required Notice: Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE.md)
# Commercial licensing inquiries: graybenjamin217@gmail.com

"""Core finite-stage operator constructions and exact analytical invariants.

Implements the Hermitian arithmetic phase matrix T_k^Herm, prime channel
decompositions C_p and C_tilde_p, 2-torsion symmetry group G_2(N_k)
desymmetrization, hybrid dilation spectrum D_k, trace moment expansions,
and canonical CRT fiber embeddings defined in:

    Benjamin Edward Gray,
    "Spectral Theory of the Primorial-Modulated Dilation Operator on
    Continuous-Discrete Fiber Spaces" (2026).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

PRIMES: Tuple[int, ...] = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def primorial(k: int) -> Tuple[int, List[int]]:
    """Return the k-th primorial N_k = p_k# and the list of its prime factors."""
    if not isinstance(k, int) or k < 1 or k > len(PRIMES):
        raise ValueError(f"Stage k must be an integer between 1 and {len(PRIMES)}.")
    chosen = list(PRIMES[:k])
    nk = 1
    for p in chosen:
        nk *= p
    return nk, chosen


def euler_totient_primorial(primes: Sequence[int]) -> int:
    """Compute Euler's totient phi(N_k) = prod_{p | N_k} (p - 1)."""
    phi = 1
    for p in primes:
        phi *= p - 1
    return phi


def exact_nullity(k: int) -> int:
    """Exact kernel dimension of T_k^Herm (Theorem 3.1)."""
    nk, primes = primorial(k)
    if k == 1:
        return 1
    phi_nk = euler_totient_primorial(primes)
    return (nk + phi_nk) // 2 - 1


def exact_rank(k: int) -> int:
    """Exact active rank of T_k^Herm (Theorem 3.1)."""
    nk, _ = primorial(k)
    return nk - exact_nullity(k)


def exact_trace(k: int) -> int:
    """Exact binary Gauss-sum trace Tr(T_k^Herm) in {0, 1} (Theorem 3.3)."""
    nk, _ = primorial(k)
    return 1 if (nk // 2) % 4 == 1 else 0


def exact_channel_inner_product(k: int, p: int, q: int) -> int:
    """Exact integer inner product (1 / N_k) Tr(C_p C_q) (Theorem 3.4)."""
    nk, primes = primorial(k)
    if p not in primes or q not in primes:
        raise ValueError(f"Primes p={p}, q={q} must divide N_{k}={nk}.")
    val = math.gcd(abs(p - q), nk) + math.gcd(p + q, nk)
    assert val % 2 == 0
    return val // 2


def exact_second_moment(k: int) -> float:
    """Exact closed-form second spectral moment Tr((T_k^Herm)^2) (Corollary 3.7)."""
    nk, primes = primorial(k)
    diag = nk / 4.0 + 0.5
    for p in primes:
        if p >= 3:
            diag += nk / (2.0 * p) + 1.0
    cross = 0.0
    for i, p in enumerate(primes):
        for q in primes[i + 1 :]:
            cross += (math.gcd(abs(p - q), nk) + math.gcd(p + q, nk)) / math.sqrt(p * q)
    return diag + cross


def build_prime_channel(k: int, p: int) -> np.ndarray:
    """Construct the unweighted real-symmetric prime channel matrix C_p."""
    nk, primes = primorial(k)
    if p not in primes:
        raise ValueError(f"Prime p={p} does not divide N_{k}={nk}.")
    idx = np.arange(nk, dtype=np.int64)
    ab_mod = np.outer(idx, idx) % nk
    return np.cos((2.0 * np.pi * ((ab_mod * p) % nk)) / nk)


def build_normalized_channel(k: int, p: int) -> np.ndarray:
    """Construct the normalized partial-symmetry channel C_tilde_p = C_p / sqrt(p N_k)."""
    nk, _ = primorial(k)
    return build_prime_channel(k, p) / math.sqrt(p * nk)


def build_T_herm(k: int) -> Tuple[np.ndarray, int, List[int]]:
    """Construct the real-symmetric arithmetic phase matrix T_k^Herm."""
    nk, primes = primorial(k)
    idx = np.arange(nk, dtype=np.int64)
    ab_mod = np.outer(idx, idx) % nk
    t_mat = np.zeros((nk, nk), dtype=np.float64)
    for p in primes:
        t_mat += (1.0 / math.sqrt(p)) * np.cos((2.0 * np.pi * ((ab_mod * p) % nk)) / nk)
    return t_mat / math.sqrt(nk), nk, primes


def build_principal_block_kronecker(k: int) -> np.ndarray:
    """Exact 2^k x 2^k matrix of T_k^Herm restricted to V_{chi_0} (Proposition 3.4)."""
    _, primes = primorial(k)
    n = 1 << k
    t_mat = np.zeros((n, n), dtype=np.float64)
    rs, ps = [], []
    for p in primes:
        s = math.sqrt(p - 1.0)
        rs.append(np.array([[-1.0, s], [s, 1.0]], dtype=np.float64) / math.sqrt(p))
        ps.append(np.array([[p - 1.0, s], [s, 1.0]], dtype=np.float64) / p)
    for j in range(k):
        mat = np.array([[1.0]], dtype=np.float64)
        for i in range(k):
            mat = np.kron(mat, ps[i] if i == j else rs[i])
        t_mat += mat
    return t_mat


def g2_generators(k: int) -> Tuple[List[int], int]:
    """Return the k - 1 CRT involutions generating G_2(N_k) for odd primes p_j | N_k."""
    nk, primes = primorial(k)
    gens: List[int] = []
    for p in primes[1:]:
        m = nk // p
        # Unique residue g mod N_k with g == -1 (mod p) and g == 1 (mod N_k / p)
        g = (1 - 2 * m * pow(m, -1, p)) % nk
        gens.append(g)
    return gens, nk


def g2_sector_projector(k: int, signs: Sequence[int]) -> np.ndarray:
    """Construct the orthogonal projector onto a single irreducible G_2(N_k) character sector."""
    gens, nk = g2_generators(k)
    if k == 1:
        if len(signs) != 0:
            raise ValueError("For k=1, G_2(2) is trivial so signs must be empty.")
        return np.eye(nk, dtype=np.float64)
    if len(signs) != len(gens):
        raise ValueError(f"Expected {len(gens)} signs for k={k}, got {len(signs)}.")
    if not all(s in (-1, 1) for s in signs):
        raise ValueError("All signs must be +1 or -1.")

    proj = np.eye(nk, dtype=np.float64)
    idx = np.arange(nk, dtype=np.int64)
    for g, s in zip(gens, signs):
        perm = (g * idx) % nk
        # Right-multiply proj by 0.5 * (I + s * U_g), where (U_g v)_a = v_{ga mod N_k}
        proj = 0.5 * (proj + s * proj[:, perm])
    return proj


def active_sector_eigenvalues(
    k: int,
    signs: Sequence[int] | None = None,
    tol: float = 1e-10,
    T_herm: np.ndarray | None = None,
) -> np.ndarray:
    """Extract non-zero eigenvalues of T_k^Herm within a single U_{-1}=+1 G_2(N_k) sector."""
    if T_herm is None:
        T_herm, _, _ = build_T_herm(k)
    if k == 1:
        mu = np.linalg.eigvalsh(T_herm)
        return mu[np.abs(mu) > tol]

    gens, _ = g2_generators(k)
    if signs is None:
        signs = tuple(1 for _ in gens)
    if len(signs) != len(gens) or int(np.prod(signs)) != 1:
        raise ValueError("Signs must have length k-1 with product +1 (U_{-1} = +1).")

    proj = g2_sector_projector(k, signs)
    eigvals_p, eigvecs_p = np.linalg.eigh(proj)
    basis_q = eigvecs_p[:, eigvals_p > 0.5]
    t_sector = basis_q.T @ T_herm @ basis_q
    mu_sector = np.linalg.eigvalsh(t_sector)
    return mu_sector[np.abs(mu_sector) > tol]


def stage_diagnostics(k: int, tol: float = 1e-10) -> Dict[str, float | int]:
    """Compute exact and numerical spectral diagnostics for Table 1 at stage k."""
    t_herm, nk, primes = build_T_herm(k)
    mu = np.linalg.eigvalsh(t_herm)
    spectral_radius = float(np.max(np.abs(mu)))
    numerical_rank = int(np.sum(np.abs(mu) > tol))
    numerical_nullity = nk - numerical_rank
    numerical_trace = float(np.trace(t_herm))
    sectors = 1 if k <= 2 else 2 ** (k - 2)
    return {
        "k": k,
        "p_k": primes[-1],
        "Nk": nk,
        "Lk": float(math.log(nk)),
        "spectral_radius": spectral_radius,
        "numerical_rank": numerical_rank,
        "exact_rank": exact_rank(k),
        "numerical_nullity": numerical_nullity,
        "exact_nullity": exact_nullity(k),
        "numerical_trace": numerical_trace,
        "exact_trace": exact_trace(k),
        "active_g2_sectors": sectors,
    }


def compute_spectrum(
    k: int,
    m_max: int = 50,
    epsilon0: float = 0.35,
    desymmetrize: bool = False,
    tol: float = 1e-10,
) -> np.ndarray:
    """Compute truncated hybrid eigenvalues lambda_{m,a}(epsilon0) of D_k."""
    t_herm, nk, _ = build_T_herm(k)
    mu = np.linalg.eigvalsh(t_herm)
    if desymmetrize:
        mu = mu[np.abs(mu) > tol]
    l_k = math.log(nk)
    m_vals = np.arange(-m_max, m_max + 1, dtype=np.float64)
    lambda_ma = (2.0 * np.pi * m_vals[:, None] / l_k) + epsilon0 * mu[None, :]
    return np.sort(lambda_ma.ravel())


def hilbert_schmidt_resolvent_norm_sq(
    k: int,
    z: complex,
    epsilon0: float = 0.35,
    m_max: int = 1000,
) -> float:
    """Compute truncated squared Hilbert-Schmidt norm ||R_z(D_k)||_HS^2 (Theorem 4.2)."""
    if z.imag == 0:
        raise ValueError("Spectral parameter z must lie in C \\ R (Im(z) != 0).")
    t_herm, nk, _ = build_T_herm(k)
    mu = np.linalg.eigvalsh(t_herm)
    l_k = math.log(nk)
    m_vals = np.arange(-m_max, m_max + 1, dtype=np.float64)
    lambda_ma = (2.0 * np.pi * m_vals[:, None] / l_k) + epsilon0 * mu[None, :]
    return float(np.sum(1.0 / (np.abs(lambda_ma - z) ** 2)))


def trace_moment_expansion(k: int, t: float, epsilon0: float) -> complex:
    """Closed-form O(1) + O(epsilon0) + O(epsilon0^2) expansion of Tr(exp(-i t epsilon0 T_k^Herm))."""
    nk, _ = primorial(k)
    m1 = exact_trace(k)
    m2 = exact_second_moment(k)
    return complex(nk - 0.5 * (t**2) * (epsilon0**2) * m2, -t * epsilon0 * m1)


def crt_fiber_injection(k: int, v: np.ndarray) -> np.ndarray:
    """Canonical CRT isometric fiber injection V_fiber: C^{N_k} -> C^{N_{k+1}}.

    Under the CRT isomorphism Z_{N_{k+1}} -> Z_{N_k} x Z_{p_{k+1}},
    x |-> (x mod N_k, x mod p_{k+1}), the uniform vector u_{p_{k+1}} maps v to:
        (V_fiber v)_x = v_{x mod N_k} / sqrt(p_{k+1}).
    """
    nk, _ = primorial(k)
    nk1, primes1 = primorial(k + 1)
    p_next = primes1[-1]
    v_arr = np.asarray(v)
    if v_arr.shape != (nk,):
        raise ValueError(f"Expected vector of length N_{k}={nk}, got shape {v_arr.shape}.")
    idx_next = np.arange(nk1, dtype=np.int64)
    return v_arr[idx_next % nk] / math.sqrt(p_next)
