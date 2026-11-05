# Improving GenD deepfake detection with score ensembling and temporal modeling

Final report of the capstone team. All numbers come from saved predictions. Scripts: `GenD/scripts/evaluate.py`, `compare_runs.py`, `ensemble.py` (ensembling) and `temporal/train_pe_gru.py`, `eval_celebdf_gru.py`, `compare_all.py` (temporal work).

## 1. Summary

GenD is a deepfake detector that keeps a large pretrained image model frozen (CLIP, Perception Encoder or DINOv3), trains only its LayerNorm layers plus a linear head, and scores a video by averaging per-frame fake probabilities. We asked whether two additions improve it: **(a)** averaging the scores of two backbones (CLIP and PE), and **(b)** a small temporal model (GRU) that reads a sequence of frames instead of averaging them independently.

What we found:
- **Averaging CLIP and PE scores gives a real improvement on unseen data.** On Celeb-DF v2 it raises AUROC from 95.4 / 95.9 to **97.1**, with intervals that exclude zero.
- **The temporal GRU is excellent in-domain but does not generalize.** On FaceForensics++ it reaches about 0.999–1.000 AUROC, but on Celeb-DF it is tied with or worse than the released models (0.946 / 0.944).
- **Merging both ideas (average of CLIP-GRU and PE-GRU) gives 0.964 on Celeb-DF.** That beats each GRU alone but is below the plain ensemble of released models (0.971).

## 2. Background

- **Backbones.** CLIP ViT-L/14 and PE-Core-L are large image encoders. In GenD they are frozen; only 0.03% of parameters are trained. Each face crop becomes a 1,024-number embedding, and a linear head turns it into a real/fake probability.
- **Original video score.** Probabilities of the frames of a video are averaged. There is no model of time.
- **Metrics.** AUROC (ranking quality, threshold-free), average precision, equal error rate, and the share of fakes caught when only 1% or 5% of real videos may be flagged (recall at fixed FPR). The 0.5 threshold is miscalibrated on Celeb-DF, so we rely on threshold-free metrics.

## 3. Datasets and protocol

| Dataset | Role | Size |
|---|---|---|
| FaceForensics++ (Deepfakes, c23) | Training and in-domain testing for the GRUs. Restricted data, kept private and never committed. | 1,020 videos in 255 groups; split 204 / 26 / 25 groups for train / validation / test; test = 100 videos (50 fake) |
| Celeb-DF v2 official test set | Cross-dataset test. Nothing was trained on it. | 518 videos (340 fake, 178 real), 32 evenly spaced face crops each (16,562 crops) |

All test sets were evaluated once, after validation, and never used to choose settings. Intervals are paired bootstrap over videos (2,000 resamples); a difference counts as real only when its interval excludes zero.

## 4. Methods we tried

### 4.1 Reproducing the original method
We ran the released GenD-CLIP and GenD-PE checkpoints on the Celeb-DF test set with the paper's test protocol (32 evenly spaced frames, mean of frame probabilities). This became the baseline: **95.4 AUROC (CLIP) and 95.9 (PE)**.

### 4.2 Score-level ensembles (no training)
We combined the two released models' saved predictions three ways:

| Method | Idea | AUROC | EER | Fakes caught at 1% FPR |
|---|---|---|---|---|
| Mean probability | Average the two probabilities per frame | 97.1 | 9.2 | 56.2 |
| Mean logit | Average log-odds, convert back | 97.1 | 8.5 | 48.2 |
| Mean rank | Average each model's percentile rank | 97.1 | 8.6 | 54.6 |

All three tie at 97.1 AUROC. A paired bootstrap shows gains of +1.7 [+0.7, +3.0] over CLIP and +1.2 [+0.5, +2.0] over PE for mean probability, so the gain is not luck. The two models agree on the overall ordering (Spearman 0.84) but make different mistakes within real and within fake videos (correlation about 0.63), which is why averaging helps. We chose **mean probability** as the default: simplest, tied for best AUROC, and best at the strict 1% false-alarm operating point.

### 4.3 Combining embeddings instead of scores (considered, not adopted)
We considered feeding concatenated CLIP and PE embeddings into a single temporal model. We did not do this because our ensemble works on scores, the two backbones may prefer different sequence lengths, and the score-level route lets each backbone keep its own trained GRU. A single concatenated-feature GRU remains possible future work.

### 4.4 Temporal GRU on CLIP features (our teammate's work)
A GRU (1,024 input, 128 hidden units, one linear output, about 443 thousand parameters) reads the sequence of per-frame CLIP embeddings. Training: Adam, learning rate 1e-3, batch 4, 20 epochs, seeds 0–4, fixed group-wise split (seed 2026) so no identity appears in two splits. The model checkpoint is chosen by best validation AUROC. A 16-frame pilot came first; the full experiment compared **32 and 64 frames per video** with five seeds each, and the 32-frame configuration was selected on validation AUROC. FF++ test AUROC was 0.99968 (32 frames) and 0.99984 (64 frames).

### 4.5 Temporal GRU on PE features (our extension)
To pair a GRU with the second backbone we:
1. **Chose 32 frames** to match the selected CLIP configuration, halve the extraction cost, and reuse the 32 crops already available for Celeb-DF.
2. **Aligned frames.** The PE extractor reads exactly the frame IDs the CLIP GRU-32 reads, so both GRUs see the same moments of each video. Only the frame positions are shared; PE computes its own features.
3. **Extracted frozen PE features** for all 1,020 FF++ videos (0 failures). We decoded each video in a single pass and detected faces only at the needed frames. Extraction ran on the CPU for face detection and took about 12.8 hours; PE encoding itself was about 20 minutes.
4. **Trained the same GRU**, same split, optimizer, loss, batch size, epochs and checkpoint rule, five seeds, changing only the input features.
5. **Merged by score averaging**: the CLIP-GRU and PE-GRU probabilities are averaged per video.

For Celeb-DF, both backbones encoded the same 32 crops per video. Thirteen videos had 30–31 crops (face detection misses), so their last crop was repeated to reach 32.

## 5. Results

### 5.1 FaceForensics++ test (100 videos, in-domain)

| Model | AUROC [95% CI] | AP | EER | Fakes caught at 1% FPR | at 5% FPR |
|---|---|---|---|---|---|
| Released GenD-CLIP | 0.9972 [0.989, 1.000] | 0.998 | 1.0% | 98% | 98% |
| Released GenD-PE | 0.9952 [0.983, 1.000] | 0.996 | 1.0% | 98% | 98% |
| Ensemble of released heads | 0.9972 [0.990, 1.000] | 0.998 | 1.0% | 98% | 98% |
| CLIP-GRU-32 | 1.0000 [1.000, 1.000] | 1.000 | 0.0% | 100% | 100% |
| PE-GRU-32 | 0.9992 [0.996, 1.000] | 0.999 | 3.0% | 98% | 100% |
| Ensemble of GRUs | 0.9996 [0.998, 1.000] | 1.000 | 2.0% | 98% | 100% |

The GRUs are at or slightly above the released heads, but all difference intervals touch zero (for example CLIP-GRU vs released CLIP: +0.0028 [0.000, +0.011]). In this setting everything is near the ceiling and 100 test videos cannot separate the methods. The GRU ensemble does not beat CLIP-GRU alone. (CLIP-GRU shows 1.0000 because averaging its five seeds ranks every test video correctly; the per-seed mean is 0.99968.) On validation the PE-GRU also matched the CLIP-GRU (0.99963 vs 0.99968), with lower calibration error than the released PE head (BCE 0.036 vs 0.071).

### 5.2 Celeb-DF v2 test (518 videos, cross-dataset)

| Model | AUROC [95% CI] | AP | EER | Fakes caught at 1% FPR | at 5% FPR |
|---|---|---|---|---|---|
| Released GenD-CLIP | 0.9541 [0.935, 0.970] | 0.972 | 11.9% | 48.2% | 80.6% |
| Released GenD-PE | 0.9590 [0.939, 0.975] | 0.975 | 10.6% | 27.4% | 78.2% |
| **Ensemble of released heads** | **0.9712** [0.955, 0.984] | 0.982 | 9.2% | **56.2%** | 84.1% |
| CLIP-GRU-32 | 0.9463 [0.924, 0.966] | 0.966 | 10.1% | 17.9% | 80.6% |
| PE-GRU-32 | 0.9438 [0.920, 0.963] | 0.965 | 12.2% | 33.8% | 66.5% |
| Ensemble of GRUs | 0.9640 [0.945, 0.979] | 0.976 | 10.8% | 28.2% | 82.4% |

| AUROC difference | Value [95% CI] | Verdict |
|---|---|---|
| Ensemble of heads vs released CLIP | +0.0172 [+0.007, +0.030] | Real gain |
| Ensemble of heads vs released PE | +0.0122 [+0.005, +0.020] | Real gain |
| Ensemble of GRUs vs CLIP-GRU | +0.0177 [+0.007, +0.030] | Real gain |
| Ensemble of GRUs vs PE-GRU | +0.0202 [+0.008, +0.034] | Real gain |
| CLIP-GRU vs released CLIP | -0.0078 [-0.022, +0.005] | Tied |
| PE-GRU vs released PE | -0.0151 [-0.025, -0.007] | Worse |
| Ensemble of GRUs vs ensemble of heads | -0.0072 [-0.016, -0.0001] | Borderline, worse |

## 6. Discussion: did our methods improve on the original?

| Question | FF++ (in-domain) | Celeb-DF (cross-dataset) |
|---|---|---|
| GRU vs the released model | Slightly better, not proven | No gain (CLIP tied, PE worse) |
| Two backbones averaged vs one | No difference (ceiling) | **Yes, +1.2 to +2.0 AUROC points, for heads and GRUs** |
| Merged method (GRU ensemble) vs original | Slightly better, not proven | Better than each GRU, below the head ensemble |
| Best overall | Tie | **Ensemble of released heads (0.971)** |

**Why the ensemble generalizes and the GRU does not.** Averaging two backbones cancels each backbone's independent errors, an effect that does not depend on which fakes appear. The GRU, in contrast, is trained on a single manipulation type (FF++ Deepfakes) and can learn that generator's temporal signature. Celeb-DF fakes are produced differently, so that signature no longer applies, while the released GenD heads were trained on broader data. This is our explanation; we did not run an experiment that isolates it.

**Practical points.** The ensemble also performs best at the strict 1% false-alarm operating point (56% vs 28% of fakes caught for the GRU ensemble), and it narrows the gap in missed fakes between women's and men's videos (PE alone missed 18.0% of fakes of women versus 9.1% of men; the ensemble reduces the first to 9.8–13.1%; only 61 fake videos of women, so this is suggestive, not proven). The cost is two forward passes per face crop.

## 7. Limitations

- The FF++ test set is only 100 videos and cannot separate methods above 99.5% AUROC.
- "Original" means the released GenD checkpoints scored by us; we did not use the paper's printed numbers.
- Celeb-DF crops keep their native size, while the FF++ training crops were 224×224 aligned faces. The backbones resize internally; we did not measure the effect.
- 13 of 518 Celeb-DF videos used a repeated last crop (30–31 real crops).
- We tested one GRU configuration trained only on FF++ Deepfakes. Training on more manipulation types was not tried, and DINOv3 was not added because its weights were not accessible to us.
- One FF++ crop (real video 029) was missing from the CLIP feature set; it lies in the training split, so validation and test were unaffected.
- FF++ is restricted data. It stays out of git and out of any shared repository.

## 8. Conclusion and future work

The most reliable improvement we found is the **score-level ensemble of GenD-CLIP and GenD-PE**: no training, statistically solid gains on an unseen dataset, and the best operating point at low false-alarm rates. The temporal GRU is a strong in-domain model but, trained only on FF++ Deepfakes, does not beat the original on Celeb-DF. Averaging the two GRUs recovers most, but not all, of the gap.

Next steps: train the GRU on several manipulation types (for example all FF++ methods) and re-test on Celeb-DF; add a third backbone (DINOv3) to the ensemble; test on further benchmarks; try a single GRU on concatenated CLIP and PE features.

## Files

- `temporal/results_pe/final_comparison_ffpp.csv`, `final_comparison_celebdf.csv`, `final_comparison_gains.csv`
- `temporal/results_pe/celebdf_gru_predictions.csv`, `gru32_pe_test_predictions.csv`
- `GenD/results/baseline/`, `GenD/results/ensemble/` (original models and score ensembles)
- `temporal/models_pe/gru32_pe_seed{0-4}_best.pt` (git-ignored)
