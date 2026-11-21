# Primorial-Modulated Dilation Operator

Reference Python implementation and numerical validation suite for:

> **Benjamin Edward Gray (2026)**, *Spectral Theory of the Primorial-Modulated Dilation Operator on Continuous-Discrete Fiber Spaces*.

This repository provides a minimal, self-contained verification codebase scoped strictly to the theorems, propositions, corollaries, and numerical tables established in the manuscript.

## Repository Structure

```text
primorial-dilation-operator/
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
├── scripts/
│   └── reproduce_table1.py       # Reproduces Table 1 (k = 1..5) and asserts exact invariants
├── src/
│   └── primorial_dilation/
│       ├── __init__.py
│       └── operator.py           # Core operator T_k^Herm, C_p, G_2(N_k), D_k, and CRT embeddings
└── tests/
    └── test_manuscript.py        # Verification suite (unittest & pytest compatible) for Sections II–VII
```

## Manuscript Verification Map

| Manuscript Result | Verification Test (`tests/test_manuscript.py`) |
| :--- | :--- |
| **Section II.B & Table 1** — Symmetry, channel sum, $k=1\dots 5$ diagnostics | `test_table_1_and_symmetry` |
| **Theorem 3.1** — Coprime sieve nullity $\dim\ker(T_k^{\mathrm{Herm}}) = \frac{N_k+\varphi(N_k)}{2}-1$ ($k \ge 2$) & explicit DFT kernel basis | `test_theorem_3_1_coprime_sieve_nullspace` |
| **Proposition 3.2** — $G_2(N_k) \cong (\mathbb{Z}/2\mathbb{Z})^{k-1}$ commutation & exact $2^{k-2}$ active parity sectors | `test_proposition_3_2_g2_symmetry_and_sector_decomposition` |
| **Theorem 3.3** — Binary Gauss-sum trace $\mathrm{Tr}(T_k^{\mathrm{Herm}}) \in \{0, 1\}$ | `test_theorem_3_3_binary_gauss_sum_trace` |
| **Theorem 3.4, Corollary 3.5, Proposition 3.6, Corollary 3.7** — Integer channel inner products, $\tilde{C}_p^3 = \tilde{C}_p$, $\Vert{}T_k^{\mathrm{Herm}}\Vert{}_2 \le k$, exact second moment | `test_theorems_3_4_to_3_7_channel_norms_and_partial_symmetry`, `test_corollary_3_7_stage_5_second_moment` |
| **Theorems 4.1 & 4.2** — Hybrid spectrum $\lambda_{m,a}(\epsilon_0)$ & Hilbert-Schmidt resolvent convergence | `test_theorems_4_1_and_4_2_hybrid_spectrum_and_hs_resolvent` |
| **Section V** — Distributional trace formula $O(1), O(\epsilon_0), O(\epsilon_0^2)$ moment expansion | `test_section_5_trace_formula_small_coupling_moments` |
| **Section VI & Proposition 6.1** — Isometry of $V_{\mathrm{fiber}}$ & explicit failure of naive spectral inheritance ($k=1\to 2, 2\to 3$) | `test_proposition_6_1_isometry_and_failure_of_naive_inheritance` |
| **Section VII.A** — Finite-stage spectral triple bounded commutator check | `test_section_7_spectral_triple_bounded_commutator` |

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run the full manuscript validation suite (standard library `unittest`, or `pytest` if installed):

```bash
python -m unittest discover -s tests -v
```

Reproduce Table 1 ($k = 1 \dots 5$):

```bash
python scripts/reproduce_table1.py
```

## Licensing

- Source code in this repository is licensed under the **MIT License** (see `LICENSE`).
- The accompanying academic manuscript is licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
