# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright (c) 2026 Benjamin Edward Gray (https://doi.org/10.5281/zenodo.22899830)
# Required Notice: Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE.md)
# Commercial licensing inquiries: graybenjamin217@gmail.com

"""Reproduce Table 1 from the manuscript and verify exact analytical invariants."""

from primorial_dilation import stage_diagnostics


def main() -> None:
    header = (
        f"| {'k':>2} | {'p_k':>3} | {'N_k':>4} | {'L_k = ln N_k':>12} | "
        f"{'||T_k^Herm||_2':>14} | {'Rank':>4} | {'Nullity':>7} | "
        f"{'Trace':>5} | {'G_2 Sectors':>11} |"
    )
    sep = (
        "|" + "-" * 4 + "|" + "-" * 5 + "|" + "-" * 6 + "|" + "-" * 14 + "|"
        + "-" * 16 + "|" + "-" * 6 + "|" + "-" * 9 + "|" + "-" * 7 + "|" + "-" * 13 + "|"
    )
    print(header)
    print(sep)
    for k in range(1, 6):
        d = stage_diagnostics(k)
        assert d["numerical_rank"] == d["exact_rank"], f"Rank mismatch at k={k}"
        assert d["numerical_nullity"] == d["exact_nullity"], f"Nullity mismatch at k={k}"
        assert abs(d["numerical_trace"] - d["exact_trace"]) < 1e-10, f"Trace mismatch at k={k}"
        print(
            f"| {d['k']:>2} | {d['p_k']:>3} | {d['Nk']:>4} | {d['Lk']:>12.4f} | "
            f"{d['spectral_radius']:>14.4f} | {d['numerical_rank']:>4} | "
            f"{d['numerical_nullity']:>7} | {d['exact_trace']:>5} | "
            f"{d['active_g2_sectors']:>11} |"
        )


if __name__ == "__main__":
    main()
