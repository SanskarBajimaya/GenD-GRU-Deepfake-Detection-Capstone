# Local-only DVC setup

## Completed migration (2026-11-05)

The proposed cleanup tree removes 1,020 FaceForensics++ videos, 32 extracted FF/Celeb-DF face samples, 20 pilot feature tensors and the legacy pilot checkpoint from Git tracking while preserving their local paths and bytes. No commit or history rewrite has been made. Prior Git history still contains these assets. See [the audit](DATASET_CLEANUP_AUDIT.md) and [separate history plan](GIT_HISTORY_CLEANUP_PLAN.md).

The existing DVC repository was reused, not initialized again. `data/faceforensic.dvc` records the unchanged `data/faceforensic` target: **1,020 files, 2,007,533,630 bytes**, directory MD5 `3b52cbd270a83c2a1ba0357bfb88411e.dir`. Root ignore rules admit `data/README.md` and DVC receipts while excluding private contents. The ignored cache is `.dvc/cache/files/md5`.

DVC 3.67.1 was installed in an isolated temporary environment, without altering project dependencies. The exact environment executable is machine-specific and intentionally not committed. Activate that environment (or install the same DVC version in your own dedicated environment) before running the commands below. The normal project Python does not currently have DVC installed. The Windows sandbox requires permission to access DVC's application/state directories; a sandbox database denial is not evidence of dataset corruption.

```powershell
python -m dvc status
python -m dvc status data/faceforensic.dvc
python -m dvc remote list
python -m dvc config --list --show-origin
```

No remote is configured at any configuration scope. No data was uploaded. Machine-local ignored `.dvc/config.local` sets `core.analytics=false` and `cache.type=copy`; it contains no credentials or absolute paths. DVC metadata uses relative paths. The migration used `dvc add --no-relink data/faceforensic`, which populates the cache without recreating workspace links. Approximately 114 GB was available before the roughly 2 GB cache copy. No checkout, garbage collection, remote configuration, push or pull was run.

## Local layout and access

```text
data/
  README.md
  faceforensic.dvc
  faceforensic/
    original_sequences/youtube/c23/videos/
    manipulated_sequences/Deepfakes/c23/videos/
```

Obtain your own provider authorization as described in [data/README.md](../data/README.md). DVC receipts do not grant access and cannot recover data without an existing authorized cache or backup. Retain an independent protected backup; the currently verified backup is temporary on the same computer under OneDrive and is not an independent backup. The backup was left untouched during migration. DVC cache is also not an independent backup.

Keep all existing relative paths, the 255-group experiment split (204/26/25), and both manifest files unchanged. The 54 approved result/report files, source, frontend, graphs and screenshots remain in Git. Extracted face samples remain locally in `datasets/FF/` and `datasets/CDFv2/`; they have no DVC receipt in this migration.

With additional owner approval, the 20 legacy `temporal/features/` tensors and `temporal/models/best_gru.pt` were removed only from the Git index and remain locally unchanged. Existing ignore rules cover these paths. They are not part of the raw-dataset DVC receipt and no new pilot-artifact receipt was created. Their historical exposure is included in the separate remediation plan.

## Checkpoints and frontend

The frontend's temporal configurations require separately obtained authorized checkpoints at:

- `temporal/models/gru32_seed0_best.pt` through `gru32_seed4_best.pt`
- `temporal/models_pe/gru32_pe_seed0_best.pt` through `gru32_pe_seed4_best.pt`

These ten checkpoints, master features and detector ONNX weights are absent from this checkout. The legacy pilot checkpoint is not a substitute. Approved original model sources are `yermandy/GenD_CLIP_L_14` and `yermandy/GenD_PE_L`; frontend preprocessing also requires its approved detector weights. Model caches outside the repository were not inventoried or changed. Uploaded-video inference does not require FF++ training videos, but model and native video dependencies must be available. Cleanup tests do not establish full inference or training success.

## Safe future operations

Before any approved local dataset update, verify backup capacity and file hashes, ensure the target is not Git-tracked, and inspect the proposed receipt change. Reuse `--no-relink` if updating its DVC receipt. Do not run checkout, garbage collection or other destructive storage operations on the only research copy. No remote or `dvc push` is authorized by this cleanup.

```powershell
git ls-files data datasets temporal/features temporal/models
git status --short
git diff --cached --stat
git check-ignore -v .dvc/cache .dvc/config.local
```

Never force-add private data or cache objects. Inspect every staged addition for data, credentials and machine-specific paths. Before the deletion commit is merged, warn collaborators and require their local backups: applying a Git deletion can remove previously tracked working files. Obtain owner review before any commit, push or history remediation.
