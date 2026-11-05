# Local research datasets

Raw FaceForensics++ videos and extracted FaceForensics++/Celeb-DF face samples are intentionally excluded from the proposed Git tree because dataset access and redistribution are subject to the original providers' restrictions. Existing Git history still contains previously committed data; see [the separate remediation plan](../docs/GIT_HISTORY_CLEANUP_PLAN.md).

Authorized researchers must obtain their own approval and follow the download process in the [official FaceForensics++ repository](https://github.com/ondyari/FaceForensics). Do not redistribute dataset files through GitHub or a public DVC remote.

The existing pipeline expects these unchanged relative paths:

```text
data/faceforensic/original_sequences/youtube/c23/videos/
data/faceforensic/manipulated_sequences/Deepfakes/c23/videos/
```

Original videos use `<source_id>.mp4`. Manipulated videos use `<source_id>_<paired_id>.mp4`; reciprocal pair members use both ID orders. Preserve the actual IDs and labels from `temporal/manifests/video_manifest.json`, rather than inferring new experiment definitions from filenames.

The historical experiment uses 510 real and 510 fake videos, organized into 255 paired groups: 204 training, 26 validation, and 25 test groups. The saved manifest and split are authoritative and must remain unchanged.

`faceforensic.dvc`, when present, is a content receipt for locally authorized data. It neither grants dataset access nor supplies the files. The ignored `.dvc/cache` is local storage, not an independent backup. No DVC remote is configured. Dataset frame samples remain locally under `datasets/FF/` and `datasets/CDFv2/`.

Back up previously tracked data before pulling or checking out the cleanup commit: Git can remove those files from a collaborator's working tree when applying their deletion from tracking. Keep the backup outside any checkout operation.
