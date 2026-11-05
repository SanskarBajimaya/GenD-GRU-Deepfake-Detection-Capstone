# Controlled GenD / GRU experiment

Implementation verification on 2026-10-13: pairing and saved split passed;
contract checks, compilation, and Git whitespace checks passed; CUDA smoke
preprocessing completed eight videos from TRAIN groups `117_217` and `226_491`
with zero failures. Actual master validation and a temporary optimizer step plus
32/64 validation evaluation passed. Eight files occupy 2,122,832 bytes total.
The preserved pilot split/dataset tests passed, and its checkpoint reproduced
the four expected probabilities (0.0179, 0.0337, 0.9492, 0.8970), with 4/4 correct.
An early `--test` attempt was rejected before constructing a TEST loader.
Full preprocessing and all ten experiment training runs were not started.

The working pilot remains a separate 16-frame experiment. No existing pilot
feature tensor or `temporal/models/best_gru.pt` is overwritten. GenD source,
pretrained weights, raw videos, `model_a.py`, `video_loader.py`,
`extract_features.py`, `model_a_pipeline.py`, `feature_dataset.py`,
`dataset_split.py`, `train_gru.py`, and `test_gru.py` remain unchanged.

## Inspection findings

* `build_feature_dataset.py`: sequence constant 16; bare tensor output; skip by
  file existence; old plural paths. Preserve as the pilot builder; corrected
  paths now use central `DATA_ROOT`.
* `feature_dataset.py`: default expected length 16, pilot feature directory;
  shape validation assumes the requested length. Preserve unchanged.
* `train_gru.py`: dataset length 16 and checkpoint length 16; five fixed pilot
  groups; selection by validation loss; seed 42; batch size 4. Preserve unchanged.
* `test_gru.py`: expects 16-frame pilot data and pilot checkpoint. Preserve.
* `compare_gend_vs_gru.py`: constant 16, `[16,3,224,224]` images, `[16,2] logits,
  checkpoint length 16, four exact frame-ID sequences and pilot probabilities.
  Preserve those checks and replace only the plural path with `DATA_ROOT`.
* `video_loader.py`: default length 16 and comments referring to 16; actual
  collection and sorting already use the supplied length. Reuse unchanged.
* `test_feature_dataset.py`, `test_dataset_split.py`: explicit expected length 16.
* `test_gend_features.py`, `test_gru_forward.py`, `test_video_loader.py`: sequence
  constant 16 and plural video paths. Only paths corrected.
* `model_a.py`: example documents 16 frames, but GRU architecture is already
  sequence-length independent. `model_a_pipeline.py` uses runtime dimensions.
* Plural references were in the builder, comparison, and the three video-based
  pilot tests above. No new large-experiment code uses that spelling.

## Pairing and reproducibility

Fake names must be three-digit IDs `A_B.mp4`, with A different from B. Sort
the two IDs to form one group ID. Require A, B, A_B, and B_A; reject any original
used in another group, unused originals, missing reciprocal fakes, and incorrect
counts. The manifest stores labels, paths, both source IDs, and all group members.
Dataset paths are relative to the repository where possible for GPU portability.

Sorted group IDs are shuffled with a private `random.Random(2026)` instance,
then divided into 204 / 26 / 25 groups. Existing manifest/split files are checked
and reused, never silently overwritten. A SHA-256 manifest fingerprint binds the
split. This prevents source leakage between the three splits. Training never
constructs a test loader; its preprocessing preflight only validates artifacts.

Five training seeds 0 through 4 seed Python, NumPy, PyTorch, all CUDA devices,
and the train DataLoader generator. Deterministic algorithms are required,
cuDNN benchmarking is disabled, and cuBLAS workspace configuration is set.
Both lengths use the unchanged 1024 -> GRU(128, one layer, dropout 0) -> Linear(1),
Adam at 1e-3, batch size 4, BCEWithLogitsLoss, and 20 epochs.

For each seed, maximize validation AUROC, then minimize validation BCE for
exact AUROC ties. Ten unique `gru{32|64}_seed{0..4}_best.pt` checkpoints contain
model state, architecture, seeds, best epoch and metrics, manifest/split hashes,
and training settings. Incomplete runs cannot enter final comparison. Existing
checkpoints cause training to stop rather than overwrite a prior run.

## Master preprocessing

The existing loader searches usable faces throughout the video in its
max-spread order, chooses the largest face, applies landmark alignment and
1.3 margin at 224x224, then sorts by original frame ID. Failure to collect 64
unique chronological frames is reported, never padded. Usable-frame counts and
error reasons go to preprocessing JSON reports. Default preprocessing scope is
the first two saved TRAIN groups (eight videos). Only `--all` selects all videos.

RetinaFace and frozen GenD load once. Each micro-batch runs the inspected HF
`feature_extractor` once and then the original `model.forward(features)` head,
the same operations as HF `GenD.forward`. This also preserves LinearNorm input
normalization when configured. No architecture or classifier replacement occurs.
Inference runs in eval/inference mode, rejects incomplete pretrained loads,
and copies float32 outputs to CPU. Resume loads and validates every existing
artifact, including identity, model name, exact shapes, finite floating values,
and 64 strictly increasing nonnegative integer frame IDs. Saves are atomic.

Each dictionary stores `[64,1024]` features, `[64,2]` original logits, frame IDs,
video ID, group ID, label, and model name. The dataset dynamically selects either
all 64 indices or `0,2,...,62` for features, logits, and IDs together. No duplicate
32-frame files are made. GenD video probabilities average per-frame softmax fake
probabilities; predictions use probability >= 0.5.

Float32 features plus logits and int64 IDs occupy 263,168 bytes per video,
268,431,360 bytes total (about 256 MiB), plus small serialization/metadata
overhead. The GRU has 443,265 parameters, about 1.69 MiB of float32 parameters
per checkpoint, or 16.9 MiB for ten, plus metadata. No optimizer states are saved.

## Results and test gate

Training writes a per-run epoch CSV with train/validation BCE, accuracy, AUROC,
precision, recall, F1, and confusion counts. `compare_large_experiment` requires
all ten complete checkpoints and evaluates validation first. It writes the
per-run `experiment_results.csv`, `validation_summary.csv`, and
`selected_configuration.json`. Selection maximizes five-seed mean validation
AUROC, then lower mean BCE, then higher mean accuracy; an exact remaining tie
prefers 32. Standard deviations are sample standard deviations (`ddof=1`).

The separate `--test` invocation requires the previously saved decision and
unchanged checkpoint hashes before constructing any test loader. It reports all
four pre-specified configurations and marks the validation-selected GRU. Individual
seed results are retained alongside `test_summary.csv`. Validation-only reruns
preserve previous test rows. No test metric guides selection.

## Commands (PowerShell, repository root)

```powershell
python -m temporal.build_manifest
python -m temporal.verify_large_experiment
python -m temporal.build_master_features --groups 2 --batch-size 16
python -m temporal.verify_large_experiment --smoke
# Stop here to review the smoke results before expensive computation.
python -m temporal.build_master_features --all --batch-size 16
python -m temporal.verify_large_experiment --all
foreach ($length in 32,64) {
    foreach ($runSeed in 0..4) {
        python -m temporal.train_large_gru --sequence-length $length --seed $runSeed
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $length / $runSeed" }
    }
}
python -m temporal.compare_large_experiment
# Review the saved validation decision, then release held-out testing.
python -m temporal.compare_large_experiment --test
```

Copy source, saved manifests/split, raw data at the same relative paths, and
the existing RetinaFace ONNX file to the college GPU environment. GenD must be
available in its Hugging Face cache or downloaded there. Required runtime
dependencies are those used by the pilot plus scikit-learn. Lower micro-batch
size if needed for GPU memory; keep it fixed and recorded for the final dataset.

## Limits and operational notes

* GenD pretraining overlap with these FF++ videos is unverified. This is an
  in-domain controlled temporal comparison, not a cross-dataset claim.
* All 20 previously used pilot videos fall in the new TRAIN split; zero are in
  validation or test. Report prior pilot development in the capstone methods.
* Do not drop failed videos or groups silently: fix preprocessing before training.
* Tensor validation detects structural/content corruption but cannot establish
  provenance of externally edited, otherwise valid feature files. Retain one
  consistent software/model cache environment when producing the master dataset.
* Exact numeric reproducibility across different GPU/software versions is not
  guaranteed; comparison checks that saved checkpoint AUROC reproduces locally.
* Interrupted training leaves an incomplete checkpoint. Archive that specific
  checkpoint and epoch CSV before intentionally restarting its run from seed.
* The current Git index already tracks 1,020 raw videos. Ignore rules only protect
  future files. Untracking data or cleaning history needs a separate Git cleanup;
  no destructive Git action is included here.
* Features and models are ignored; source, small JSON manifests/split, and CSV
  results remain eligible for tracking. Full preprocessing and ten-run training
  are never launched automatically by the verification commands.
