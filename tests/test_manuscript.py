# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright (c) 2026 Benjamin Edward Gray (https://doi.org/10.5281/zenodo.22899830)
# Required Notice: Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE.md)
# Commercial licensing inquiries: graybenjamin217@gmail.com

"""Direct validation suite for every theorem, proposition, and table in the manuscript.

Compatible with both standard-library unittest (`python -m unittest -v`) and
pytest (`pytest -v`).

Manuscript:
    Benjamin Edward Gray (2026),
    "Spectral Theory of the Primorial-Modulated Dilation Operator on
    Continuous-Discrete Fiber Spaces."
"""

from __future__ import annotations

import itertools
import math
import unittest

import numpy as np

from primorial_dilation import (
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

# Published Table 1 reference values (Section II.D)
TABLE_1_EXPECTED = {
    1: (2, 2, 0.6931, 1.0000, 1, 1, 1, 1),
    2: (3, 6, 1.7918, 1.7703, 3, 3, 0, 1),
    3: (5, 30, 3.4012, 2.5052, 12, 18, 0, 2),
    4: (7, 210, 5.3471, 3.2055, 82, 128, 1, 4),
    5: (11, 2310, 7.7450, 3.8682, 916, 1394, 0, 8),
}


class TestManuscriptValidation(unittest.TestCase):
    """Exact verification of Sections II–VII of the manuscript."""

    def test_table_1_and_symmetry(self) -> None:
        """Section II.B & Table 1: symmetry, channel sum, and k=1..5 diagnostics."""
        for k in (1, 2, 3, 4, 5):
            with self.subTest(k=k):
                t_herm, _, primes = build_T_herm(k)
                self.assertTrue(np.allclose(t_herm, t_herm.T, atol=1e-13))

                reconstructed = sum(build_normalized_channel(k, p) for p in primes)
                self.assertTrue(np.allclose(t_herm, reconstructed, atol=1e-13))

                diag = stage_diagnostics(k)
                exp_pk, exp_nk, exp_lk, exp_rad, exp_rank, exp_null, exp_tr, exp_sec = (
                    TABLE_1_EXPECTED[k]
                )
                self.assertEqual(diag["p_k"], exp_pk)
                self.assertEqual(diag["Nk"], exp_nk)
                self.assertAlmostEqual(diag["Lk"], exp_lk, places=4)
                self.assertAlmostEqual(diag["spectral_radius"], exp_rad, places=4)
                self.assertEqual(diag["numerical_rank"], exp_rank)
                self.assertEqual(diag["exact_rank"], exp_rank)
                self.assertEqual(diag["numerical_nullity"], exp_null)
                self.assertEqual(diag["exact_nullity"], exp_null)
                self.assertAlmostEqual(diag["numerical_trace"], exp_tr, places=10)
                self.assertEqual(diag["exact_trace"], exp_tr)
                self.assertEqual(diag["active_g2_sectors"], exp_sec)

    def test_theorem_3_1_coprime_sieve_nullspace(self) -> None:
        """Theorem 3.1: exact nullity formula and explicit orthonormal DFT kernel basis."""
        for k in (1, 2, 3, 4):
            with self.subTest(k=k):
                t_herm, nk, primes = build_T_herm(k)
                mu = np.linalg.eigvalsh(t_herm)
                num_nullity = int(np.sum(np.abs(mu) <= 1e-10))
                self.assertEqual(num_nullity, exact_nullity(k))
                self.assertEqual(nk - num_nullity, exact_rank(k))

                if k == 1:
                    v_ker = np.array([1.0, -1.0]) / math.sqrt(2.0)
                    self.assertLess(float(np.linalg.norm(t_herm @ v_ker)), 1e-13)
                    continue

                b = np.arange(nk, dtype=np.float64)
                dft = np.exp(2j * np.pi * np.outer(b, b) / nk) / math.sqrt(nk)

                kernel_vectors = []
                for c in range(1, nk // 2):
                    f_minus = (dft[:, c] - dft[:, nk - c]) / math.sqrt(2.0)
                    kernel_vectors.append(f_minus)
                    if math.gcd(c, nk) == 1:
                        f_plus = (dft[:, c] + dft[:, nk - c]) / math.sqrt(2.0)
                        kernel_vectors.append(f_plus)

                phi_nk = euler_totient_primorial(primes)
                self.assertEqual(len(kernel_vectors), (nk + phi_nk) // 2 - 1)
                self.assertEqual(len(kernel_vectors), exact_nullity(k))

                K = np.column_stack(kernel_vectors)
                self.assertTrue(np.allclose(K.conj().T @ K, np.eye(K.shape[1]), atol=1e-12))
                self.assertLess(float(np.linalg.norm(t_herm @ K)), 1e-11)

    def test_proposition_3_2_g2_symmetry_and_sector_decomposition(self) -> None:
        """Proposition 3.2: G_2(N_k) generators, commutation, and exact 2^{k-2} sector union."""
        for k in (2, 3, 4):
            with self.subTest(k=k):
                t_herm, nk, primes = build_T_herm(k)
                gens, _ = g2_generators(k)
                self.assertEqual(len(gens), k - 1)

                prod_g = 1
                idx = np.arange(nk, dtype=np.int64)
                for g, p in zip(gens, primes[1:]):
                    self.assertEqual(g % p, p - 1)
                    self.assertEqual(g % (nk // p), 1)
                    self.assertEqual((g * g) % nk, 1)
                    prod_g = (prod_g * g) % nk

                    perm = (g * idx) % nk
                    self.assertTrue(np.allclose(t_herm[np.ix_(perm, perm)], t_herm, atol=1e-13))
                    for q in primes:
                        cq = build_prime_channel(k, q)
                        self.assertTrue(np.allclose(cq[np.ix_(perm, perm)], cq, atol=1e-13))

                self.assertEqual(prod_g, nk - 1)

                valid_sign_tuples = [
                    s for s in itertools.product((-1, 1), repeat=k - 1) if int(np.prod(s)) == 1
                ]
                self.assertEqual(len(valid_sign_tuples), 2 ** (k - 2))

                projectors = [g2_sector_projector(k, s) for s in valid_sign_tuples]
                for i, pi in enumerate(projectors):
                    self.assertTrue(np.allclose(pi @ pi, pi, atol=1e-12))
                    self.assertTrue(np.allclose(pi, pi.T, atol=1e-12))
                    for j in range(i + 1, len(projectors)):
                        self.assertTrue(np.allclose(pi @ projectors[j], 0.0, atol=1e-12))

                sector_eigs = np.concatenate(
                    [active_sector_eigenvalues(k, signs=s, T_herm=t_herm) for s in valid_sign_tuples]
                )
                full_eigs = np.linalg.eigvalsh(t_herm)
                full_active = np.sort(full_eigs[np.abs(full_eigs) > 1e-10])
                self.assertEqual(len(sector_eigs), exact_rank(k))
                self.assertTrue(np.allclose(np.sort(sector_eigs), full_active, atol=1e-10))

    def test_theorem_3_3_binary_gauss_sum_trace(self) -> None:
        """Theorem 3.3: Tr(T_k^Herm) in {0, 1} and odd prime channels vanish identically."""
        for k in (1, 2, 3, 4, 5):
            with self.subTest(k=k):
                t_herm, nk, primes = build_T_herm(k)
                r_k = sum(1 for p in primes if p % 4 == 3)
                expected = 1 if (r_k % 2 == 0) else 0
                self.assertEqual(expected, exact_trace(k))
                self.assertAlmostEqual(float(np.trace(t_herm)), expected, places=10)

                idx = np.arange(nk, dtype=np.int64)
                a2_mod = (idx * idx) % nk
                for p in primes:
                    tr_cp = float(np.sum(np.cos((2.0 * np.pi * ((a2_mod * p) % nk)) / nk)))
                    if p >= 3:
                        self.assertLess(abs(tr_cp), 1e-10)
                    else:
                        self.assertLess(abs(tr_cp / math.sqrt(2 * nk) - expected), 1e-10)

    def test_theorems_3_4_to_3_7_channel_norms_and_partial_symmetry(self) -> None:
        """Theorem 3.4, Corollary 3.5, Proposition 3.6, and Corollary 3.7 for k=1..4."""
        for k in (1, 2, 3, 4):
            with self.subTest(k=k):
                t_herm, nk, primes = build_T_herm(k)
                channels = {p: build_prime_channel(k, p) for p in primes}
                norm_channels = {p: channels[p] / math.sqrt(p * nk) for p in primes}

                for p in primes:
                    cp = channels[p]
                    diag_expected = (nk // 2 + 1) if p == 2 else (nk // 2 + p)
                    diag_actual = float(np.sum(cp * cp)) / nk
                    self.assertAlmostEqual(diag_actual, diag_expected, places=10)

                    for q in primes:
                        cq = channels[q]
                        inner_actual = float(np.sum(cp * cq)) / nk
                        inner_expected = exact_channel_inner_product(k, p, q)
                        self.assertAlmostEqual(inner_actual, inner_expected, places=10)

                for p in primes:
                    ct = norm_channels[p]
                    ct3 = ct @ ct @ ct
                    self.assertTrue(np.allclose(ct3, ct, atol=1e-11))
                    eigs_ct = np.linalg.eigvalsh(ct)
                    self.assertAlmostEqual(float(np.max(np.abs(eigs_ct))), 1.0, places=11)
                    dist_to_set = np.min(
                        np.abs(eigs_ct[:, None] - np.array([-1.0, 0.0, 1.0])[None, :]), axis=1
                    )
                    self.assertLess(float(np.max(dist_to_set)), 1e-11)
                    expected_ch_rank = (nk + 2) // 4 if p == 2 else nk // (2 * p) + 1
                    self.assertEqual(int(np.sum(np.abs(eigs_ct) > 1e-10)), expected_ch_rank)

                spectral_radius = float(np.max(np.abs(np.linalg.eigvalsh(t_herm))))
                self.assertLessEqual(spectral_radius, k + 1e-12)

                num_second_moment = float(np.sum(t_herm * t_herm))
                self.assertAlmostEqual(num_second_moment, exact_second_moment(k), places=9)

    def test_corollary_3_7_stage_5_second_moment(self) -> None:
        """Corollary 3.7 & Proposition 3.6 norm bound at stage k=5 (N_5=2310)."""
        t_herm, _, _ = build_T_herm(5)
        num_second_moment = float(np.sum(t_herm * t_herm))
        self.assertAlmostEqual(num_second_moment, exact_second_moment(5), places=7)
        self.assertLessEqual(float(np.max(np.abs(np.linalg.eigvalsh(t_herm)))), 5.0 + 1e-12)

    def test_theorems_4_1_and_4_2_hybrid_spectrum_and_hs_resolvent(self) -> None:
        """Theorems 4.1 & 4.2: hybrid eigenvalues lambda_{m,a} and HS resolvent convergence."""
        k = 3
        eps = 0.35
        spec_full = compute_spectrum(k, m_max=20, epsilon0=eps, desymmetrize=False)
        spec_active = compute_spectrum(k, m_max=20, epsilon0=eps, desymmetrize=True)
        nk, _ = primorial(k)
        self.assertEqual(len(spec_full), 41 * nk)
        self.assertEqual(len(spec_active), 41 * exact_rank(k))
        self.assertTrue(np.all(np.isreal(spec_full)))

        z = 0.4 + 1.2j
        hs_200 = hilbert_schmidt_resolvent_norm_sq(k, z, epsilon0=eps, m_max=200)
        hs_2000 = hilbert_schmidt_resolvent_norm_sq(k, z, epsilon0=eps, m_max=2000)
        hs_4000 = hilbert_schmidt_resolvent_norm_sq(k, z, epsilon0=eps, m_max=4000)
        self.assertLess(hs_200, hs_2000)
        self.assertLess(hs_2000, hs_4000)
        self.assertLess(hs_4000 - hs_2000, 0.05)

    def test_section_5_trace_formula_small_coupling_moments(self) -> None:
        """Section V: exact O(1), O(eps), O(eps^2) expansion of Tr(exp(-i t eps T_k^Herm))."""
        t_val = 1.3
        eps1 = 1e-2
        eps2 = 5e-3
        for k in (1, 2, 3, 4):
            with self.subTest(k=k):
                t_herm, _, _ = build_T_herm(k)
                mu = np.linalg.eigvalsh(t_herm)

                exact_tr1 = complex(np.sum(np.exp(-1j * t_val * eps1 * mu)))
                err1 = abs(exact_tr1 - trace_moment_expansion(k, t_val, eps1))

                exact_tr2 = complex(np.sum(np.exp(-1j * t_val * eps2 * mu)))
                err2 = abs(exact_tr2 - trace_moment_expansion(k, t_val, eps2))

                if err1 > 1e-12:
                    ratio = err1 / err2
                    self.assertGreater(ratio, 7.5)
                    self.assertLess(ratio, 8.5)
                else:
                    self.assertLess(err2, 1e-12)

    def test_proposition_6_1_isometry_and_failure_of_naive_inheritance(self) -> None:
        """Section VI & Proposition 6.1: V_fiber isometry and explicit non-inheritance."""
        for k in (1, 2):
            nk, _ = primorial(k)
            cols = [crt_fiber_injection(k, np.eye(nk)[:, i]) for i in range(nk)]
            V = np.column_stack(cols)
            self.assertTrue(np.allclose(V.T @ V, np.eye(nk), atol=1e-13))

        t1, _, _ = build_T_herm(1)
        t2, _, _ = build_T_herm(2)

        mu1 = np.linalg.eigvalsh(t1)
        active1 = mu1[np.abs(mu1) > 1e-10]
        self.assertTrue(np.allclose(active1, [1.0], atol=1e-13))

        u2 = np.array([1.0, 1.0]) / math.sqrt(2.0)
        self.assertTrue(np.allclose(t1 @ u2, u2, atol=1e-13))

        uplifted = crt_fiber_injection(1, u2)
        image = t2 @ uplifted
        rayleigh = float(np.dot(uplifted, image))
        residual = float(np.linalg.norm(image - rayleigh * uplifted))
        self.assertGreater(residual, 0.5)

        A2 = np.array(
            [
                [1.0 / math.sqrt(2.0) + 1.0 / math.sqrt(3.0), 1.0 / math.sqrt(2.0), math.sqrt(2.0 / 3.0)],
                [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0), 0.0],
                [math.sqrt(2.0 / 3.0), 0.0, -1.0 / math.sqrt(3.0)],
            ],
            dtype=np.float64,
        )
        eigs_A2 = np.linalg.eigvalsh(A2)
        mu2 = np.linalg.eigvalsh(t2)
        active2 = np.sort(mu2[np.abs(mu2) > 1e-10])
        self.assertTrue(np.allclose(eigs_A2, active2, atol=1e-12))

        for lam in active2:
            p_val = (
                lam**3
                - (2.0 + 1.0 / math.sqrt(6.0)) * lam
                - (1.0 / math.sqrt(2.0) + 1.0 / math.sqrt(3.0))
            )
            self.assertLess(abs(p_val), 1e-12)

        self.assertGreater(float(np.min(np.abs(active2 - 1.0))), 0.5)

        t3, _, _ = build_T_herm(3)
        eigvals2, eigvecs2 = np.linalg.eigh(t2)
        top_vec2 = eigvecs2[:, np.argmax(np.abs(eigvals2))]
        uplifted2 = crt_fiber_injection(2, top_vec2)
        img3 = t3 @ uplifted2
        res3 = float(np.linalg.norm(img3 - float(np.dot(uplifted2, img3)) * uplifted2))
        self.assertGreater(res3, 0.5)

    def test_section_7_spectral_triple_bounded_commutator(self) -> None:
        """Section VII.A: finite-stage spectral triple commutator identity and norm bound."""
        k = 3
        eps = 0.35
        t_herm, nk, _ = build_T_herm(k)
        rng = np.random.default_rng(42)
        M = rng.normal(size=(nk, nk)) + 1j * rng.normal(size=(nk, nk))
        comm_fiber = t_herm @ M - M @ t_herm
        l_k = math.log(primorial(k)[0])
        r = 2
        gp_inf = 2.0 * np.pi * abs(r) / l_k
        bound = gp_inf * float(np.linalg.norm(M, 2)) + abs(eps) * float(np.linalg.norm(comm_fiber, 2))
        exact_block = (2.0 * np.pi * r / l_k) * M + eps * comm_fiber
        self.assertLessEqual(float(np.linalg.norm(exact_block, 2)), bound + 1e-12)


if __name__ == "__main__":
    unittest.main()
