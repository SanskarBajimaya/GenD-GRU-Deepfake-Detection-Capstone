# Final in-domain TEST report

Exactly 100 previously unevaluated TEST videos (25 groups, 50 REAL and 50 FAKE) were evaluated once per model/run. All five seeds are reported independently; no ensemble was created.

## PRIMARY FINAL RESULT

GRU-32 was selected before TEST exclusively by mean validation AUROC. This decision is unchanged.

| Model | Frames | AUROC | Accuracy | BCE | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GenD | 32 | 0.997200 | 0.960000 | 0.141138 | 0.942308 | 0.980000 | 0.960784 |
| GRU | 32 | 0.999680 +/- 0.000335 | 0.990000 +/- 0.012247 | 0.031087 +/- 0.027984 | 0.991997 +/- 0.010962 | 0.988000 +/- 0.017889 | 0.989939 +/- 0.012371 |

GRU-32 minus GenD-32: AUROC +0.002480; accuracy +0.030000; BCE -0.110051. Lower BCE is better.

## Secondary 64-frame result

| Model | Frames | AUROC | Accuracy | BCE | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GenD | 64 | 0.997200 | 0.950000 | 0.142193 | 0.924528 | 0.980000 | 0.951456 |
| GRU | 64 | 0.999840 +/- 0.000219 | 0.994000 +/- 0.005477 | 0.027559 +/- 0.016692 | 0.992157 +/- 0.010740 | 0.996000 +/- 0.008944 | 0.994019 +/- 0.005460 |

GRU-64 minus GenD-64: AUROC +0.002640; accuracy +0.044000; BCE -0.114634. Lower BCE is better.

GRU-32 remains the pre-selected model. The 64-frame experiment is a secondary comparison. Standard deviations are sample standard deviations across five independent seeds (ddof=1); accuracy is a fraction.

## Individual results and confusion counts

| Model | Frames | Seed | AUROC | Accuracy | BCE | Precision | Recall | F1 | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GenD | 32 |  | 0.997200 | 0.960000 | 0.141138 | 0.942308 | 0.980000 | 0.960784 | 49 | 47 | 3 | 1 |
| GRU | 32 | 0 | 0.999600 | 0.990000 | 0.026329 | 1.000000 | 0.980000 | 0.989899 | 49 | 50 | 0 | 1 |
| GRU | 32 | 1 | 1.000000 | 1.000000 | 0.012468 | 1.000000 | 1.000000 | 1.000000 | 50 | 50 | 0 | 0 |
| GRU | 32 | 2 | 0.999600 | 0.990000 | 0.075385 | 0.980392 | 1.000000 | 0.990099 | 50 | 49 | 1 | 0 |
| GRU | 32 | 3 | 0.999200 | 0.970000 | 0.037686 | 0.979592 | 0.960000 | 0.969697 | 48 | 49 | 1 | 2 |
| GRU | 32 | 4 | 1.000000 | 1.000000 | 0.003568 | 1.000000 | 1.000000 | 1.000000 | 50 | 50 | 0 | 0 |
| GenD | 64 |  | 0.997200 | 0.950000 | 0.142193 | 0.924528 | 0.980000 | 0.951456 | 49 | 46 | 4 | 1 |
| GRU | 64 | 0 | 0.999600 | 0.990000 | 0.051801 | 0.980392 | 1.000000 | 0.990099 | 50 | 49 | 1 | 0 |
| GRU | 64 | 1 | 1.000000 | 0.990000 | 0.028883 | 0.980392 | 1.000000 | 0.990099 | 50 | 49 | 1 | 0 |
| GRU | 64 | 2 | 1.000000 | 1.000000 | 0.011397 | 1.000000 | 1.000000 | 1.000000 | 50 | 50 | 0 | 0 |
| GRU | 64 | 3 | 0.999600 | 0.990000 | 0.033335 | 1.000000 | 0.980000 | 0.989899 | 49 | 50 | 0 | 1 |
| GRU | 64 | 4 | 1.000000 | 1.000000 | 0.012378 | 1.000000 | 1.000000 | 1.000000 | 50 | 50 | 0 | 0 |

## Descriptive error analysis

No aggregate GRU prediction is defined. Shared, fixed, and new errors are therefore reported separately for each seed.

### 32 frames

GenD errors: 216_164, 452, 716, 987.

| GRU seed | GRU errors | Shared errors | GenD errors fixed | New GRU errors |
|---:|---|---|---|---|
| 0 | 216_164 | 216_164 | 452, 716, 987 | none |
| 1 | none | none | 216_164, 452, 716, 987 | none |
| 2 | 452 | 452 | 216_164, 716, 987 | none |
| 3 | 216_164, 452, 697_581 | 216_164, 452 | 716, 987 | 697_581 |
| 4 | none | none | 216_164, 452, 716, 987 | none |

GRU error counts across five seeds: 216_164: 2/5, 452: 2/5, 697_581: 1/5.

### 64 frames

GenD errors: 216_164, 452, 697, 716, 987.

| GRU seed | GRU errors | Shared errors | GenD errors fixed | New GRU errors |
|---:|---|---|---|---|
| 0 | 452 | 452 | 216_164, 697, 716, 987 | none |
| 1 | 452 | 452 | 216_164, 697, 716, 987 | none |
| 2 | none | none | 216_164, 452, 697, 716, 987 | none |
| 3 | 216_164 | 216_164 | 452, 697, 716, 987 | none |
| 4 | none | none | 216_164, 452, 697, 716, 987 | none |

GRU error counts across five seeds: 216_164: 1/5, 452: 2/5.

## Interpretation and limitations

For the pre-selected GRU-32, mean accuracy changes by +3.00 percentage points (equivalent to +3.00 correctly classified videos per 100, averaged across seeds). AUROC changes by +0.002480 and BCE by -0.110051. These are descriptive numerical differences; no statistical significance test was performed.

The five-seed means and sample standard deviations describe training-seed variability, not uncertainty across datasets or source groups. The per-seed error table shows whether individual improvements are consistent.

This is an in-domain FaceForensics++ c23 test. As specified in the experiment context, pretrained GenD was itself trained on FaceForensics++ c23; this result does not establish cross-dataset generalization. External evaluation has not been performed.

All ten checkpoint hashes and the saved validation decision were verified before TEST inference. No retraining, checkpoint modification, epoch reselection, model reselection, threshold tuning, hyperparameter tuning, preprocessing, feature regeneration, or ensembling occurred. Stored logits supply the frozen GenD baseline.

Structured outputs: test_predictions.csv, test_metrics.csv, test_summary.csv, experiment_results.csv (preserves validation rows), test_integrity.json, test_error_analysis.json, final_experiment_summary.json. The existing selected_configuration.json and checkpoints are preserved.
