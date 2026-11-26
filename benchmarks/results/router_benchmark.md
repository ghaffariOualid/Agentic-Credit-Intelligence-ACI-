# Benchmark comparatif du routeur ACI (profil GPU T4 16 Go)

- **Épisodes d'entraînement** : 120 | **Évaluation** : 40 épisodes × 3 seeds `[42, 101, 202]`

| Stratégie | Qualité (moy ± σ) | Coût ($) | Coût relatif (vs Large) | Latence moy (s) | Latence P95 (s) | Reward (moy ± σ) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Static-Small | 0.823 ± 0.007 | 0.059 | 0.14 | 1.93 | 4.58 | 0.428 ± 0.006 |
| Static-Medium | 0.940 ± 0.002 | 0.161 | 0.38 | 4.17 | 5.23 | 0.437 ± 0.001 |
| Static-Large | 0.970 ± 0.001 | 0.421 | 1.00 | 7.91 | 11.50 | 0.331 ± 0.001 |
| Bandit (LinUCB) | 0.890 ± 0.003 | 0.097 | 0.23 | 2.75 | 5.05 | 0.450 ± 0.003 |
| Supervised (PyTorch MLP) | 0.919 ± 0.030 | 0.137 | 0.33 | 3.65 | 5.08 | 0.440 ± 0.002 |
| RL (PPO + Masking) | 0.864 ± 0.055 | 0.105 | 0.25 | 2.87 | 6.56 | 0.426 ± 0.008 |
