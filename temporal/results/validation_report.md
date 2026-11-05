# GRU training and validation results

All ten sequential 20-epoch runs completed. All checkpoints passed metadata, finite-weight, best-epoch and reproduced-validation-metric checks. TEST was not loaded or evaluated during this stage.

| Model | Frames | Validation AUROC | Validation accuracy | Validation BCE |
|---|---:|---:|---:|---:|
| GenD | 32 | 0.998151 | 0.971154 | 0.157012 |
| GRU | 32 | 0.999334 +/- 0.000992 | 0.988462 +/- 0.010533 | 0.063913 +/- 0.071159 |
| GenD | 64 | 0.997781 | 0.980769 | 0.155915 |
| GRU | 64 | 0.997781 +/- 0.002108 | 0.990385 +/- 0.000000 | 0.054339 +/- 0.028475 |

GRU standard deviations use the sample standard deviation (ddof=1) across five seeds. Accuracy is a fraction.

| Checkpoint (under temporal/models/) | Best epoch | Validation AUROC | Validation accuracy | Validation BCE | Run seconds |
|---|---:|---:|---:|---:|---:|
| gru32_seed0_best.pt | 7 | 0.998891 | 0.990385 | 0.081330 | 49.9 |
| gru32_seed1_best.pt | 1 | 1.000000 | 0.990385 | 0.021484 | 49.4 |
| gru32_seed2_best.pt | 17 | 0.997781 | 0.971154 | 0.179632 | 49.0 |
| gru32_seed3_best.pt | 1 | 1.000000 | 0.990385 | 0.036532 | 49.1 |
| gru32_seed4_best.pt | 12 | 1.000000 | 1.000000 | 0.000585 | 49.1 |
| gru64_seed0_best.pt | 14 | 0.997041 | 0.990385 | 0.096782 | 51.5 |
| gru64_seed1_best.pt | 1 | 0.997411 | 0.990385 | 0.051126 | 51.3 |
| gru64_seed2_best.pt | 2 | 0.994822 | 0.990385 | 0.065832 | 51.4 |
| gru64_seed3_best.pt | 1 | 0.999630 | 0.990385 | 0.031531 | 51.2 |
| gru64_seed4_best.pt | 1 | 1.000000 | 0.990385 | 0.026424 | 51.8 |

Selected GRU = GRU-32. GRU-32 mean AUROC (0.999334) exceeds GRU-64 (0.997781), so the primary selection criterion decides; BCE and accuracy tie-breakers are not reached.

Total measured run time: 503.7 seconds (8.40 minutes). Times include per-run preflight, training, validation and epoch result writes, and exclude interpreter startup and final comparison.

Observations: several runs have rising validation BCE in later epochs; best checkpoints follow the requested AUROC/BCE rule. GRU-64 has lower mean BCE and slightly higher mean accuracy, while GRU-32 wins the primary mean-AUROC criterion. No training failures or checkpoint verification failures occurred.

Changes were limited to restricting preflight artifact reads to active splits and recording elapsed run time. Architecture, hyperparameters, frozen features, saved split and preprocessing were preserved.

Final TEST evaluation is pending explicit user confirmation.
