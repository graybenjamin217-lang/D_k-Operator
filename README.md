# D_k-Operator
primorial-dilation-operator/
├── .gitignore
├── LICENSE                         # MIT for code (CC BY 4.0 noted for manuscript)
├── README.md                       # Theorem-to-test verification map & quickstart
├── pyproject.toml
├── requirements.txt                # numpy>=1.24.0
├── scripts/
│   └── reproduce_table1.py         # Prints Table 1 (k = 1..5) and asserts exact invariants
├── src/
│   └── primorial_dilation/
│       ├── __init__.py
│       └── operator.py             # T_k^Herm, C_p, C_tilde_p, G_2(N_k), D_k, moments, V_fiber
└── tests/
    └── test_manuscript.py          # 10 verification test methods covering Sections II–VII
