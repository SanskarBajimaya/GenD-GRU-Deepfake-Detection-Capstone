# Dataset cleanup audit

Audit date: 2026-11-05. Cleanup branch base: latest fetched `origin/main`, `0b4f7451d1252bb39d2d0437b8f0951e51ae5340`. Original checkout and verified temporary backup HEAD: `499fdbbb9fef73bbf6fdb96c6807d470a7aaa42a`. No commit, push, merge, or history rewrite is part of this change.

## Classification before index changes

| Location | Files | Bytes | Class and decision |
|---|---:|---:|---|
| `data/faceforensic/original_sequences/` | 510 | 987244527 | B: untrack; retain every local video |
| `data/faceforensic/manipulated_sequences/` | 510 | 1020289103 | B: untrack; retain every local video |
| `datasets/FF/` | 20 | 1641523 | B: untrack; retain local face samples |
| `datasets/CDFv2/` | 12 | 666351 | B: untrack; retain local face samples |
| `temporal/features/` | 20 | 1341340 | C: approved local-only storage; untracked, local bytes preserved |
| `temporal/models/best_gru.pt` | 1 | 1775722 | C: approved local-only storage; untracked, local bytes preserved |
| `temporal/manifests/` | 2 | 432758 | A: preserve byte-for-byte |
| `temporal/results/`, `temporal/results_pe/` | 54 | 237201 | A: preserve byte-for-byte |
| `docs/readme-assets/` | 7 | 1245151 | A: retain graphs/screenshots |
| `media/gradio.png` | 1 | 3324286 | A: retain frontend documentation asset |
| `src/model/forada/clip/bpe_simple_vocab_16e6.txt.gz` | 1 | 1356917 | D: retain tokenizer vocabulary; not a dataset archive |
| Source, frontend and other documentation | — | — | A/D: preserve except explicitly updated cleanup documentation |

No additional local videos, image datasets, checkpoints, ONNX files, research archives, or DVC cache were found in the full working-tree inventory before migration. In particular, the ten expected five-seed CLIP/PE GRU-32 checkpoints, master feature directories, and local detector ONNX weight are absent; this cleanup does not recreate or substitute them. Earlier documentation describing a present ONNX file did not match this checkout.

## Backup and preservation evidence

The user-approved temporary backup was verified before changes. Every existing project/research file and all Git object files matched SHA-256. All 1020 raw videos, 32 face samples, 21 feature/checkpoint files, 54 result files and both manifests matched. Separate filesystem identities were confirmed. The backup remains untouched. It is on the same computer under OneDrive, not an independent protected backup. The fetched main commits postdate that backup; they remain in the current Git object database and were not written into the backup.

Pre-cleanup file SHA-256 values are recorded locally in ignored `tmp/dataset-cleanup-verification/before.json`; final comparison evidence is saved alongside it. These local verification records are not dataset distribution artifacts and are not staged.

Manifest SHA-256 values:

```text
video_manifest.json  3c6e52419499648f69f8fe317c285d9b1782f69194acea941c57688d644c9e81
split_seed_2026.json ebae2be53bb349828a932503725ca81579b9d31afe77def48a7418c84b8ae01c
```

## Historical exposure

All locally available branches, remote-tracking refs, tags (none), Codex refs, reachable object names, and historical changed paths were inspected. Raw data also appeared historically under `data/faceforensics/` (plural; 20 video paths). Current paths contain 1020 unique video names. Face sample, feature and pilot checkpoint paths also appear in history. `media/gradio.mp4` is a historical demo asset whose provenance must be reviewed before any decision to filter it. Current approved screenshots remain preserved.

This inventory is not a claim to have inspected inaccessible GitHub PR refs, caches, forks, or other clones. Removing files from the index does not purge any of those histories. See [the separate plan](GIT_HISTORY_CLEANUP_PLAN.md).

Before anyone pulls or checks out a future cleanup commit, require local data backups: Git may remove their previously tracked files. Coordinate this warning before merging; no messages have been sent to teammates automatically.

## Completed validation

- Every original file remains on disk. SHA-256 comparison confirms all 1020 videos, 32 face samples, 21 tensors/checkpoint files, both manifests, all 54 result files and all 11 frontend files are unchanged. All other baseline files match except the intentionally edited root README and `.gitignore`; files introduced by latest main are preserved except the intentionally updated DVC guide.
- The DVC receipt contains 1020 files and 2007533630 bytes. All 1021 cache objects (1020 blobs plus directory listing) passed MD5-to-object-name verification. Both `dvc status` and explicit receipt status report `Data and pipelines are up to date.` No remote is configured. Migration used `--no-relink` and local copy mode; local data was not deleted, moved or replaced.
- A broad `/data/*` ignore pattern made DVC's Windows Git backend treat the parent as ignored despite Git's receipt exceptions. Exact private directory ignore rules fix stage discovery while allowing the receipt and README to be tracked.
- `python -m temporal.verify_large_experiment`: PASS (split, corruption rejection, shared indices, synthetic backward-pass contract, encoder pass, metrics, checkpoint names). No research training or metric recalculation was performed.
- `python -m unittest discover -s frontend/tests -v`: 14 tests; 10 passed, 3 errors, 1 skipped. This matches the pre-cleanup baseline: installed Gradio rejects `Video(buttons=...)`; FFmpeg is unavailable for one integration fixture. Frontend source is unchanged. Full model inference and retraining were not verified.
- Staged changes: 1073 index deletions, four added documentation/receipt files, and three modified documentation/ignore files. No cache, local configuration, raw data addition or credentials are staged. The 20 feature tensors and pilot checkpoint are now untracked with explicit owner approval; all original local hashes are preserved. `git diff --cached --check` passes.
- No commit, push, merge, history rewrite, branch/tag deletion, DVC garbage collection or data upload was performed. The temporary backup was not changed. Current raw-data history is still present and requires separate owner-approved remediation.

## Approved pilot artifact cleanup

The owner additionally approved index-only removal of the 20 pilot tensors and legacy checkpoint on the existing cleanup branch. `git rm -r --cached -- temporal/features temporal/models/best_gru.pt` preserved every working file. Existing `/temporal/features/` and `/temporal/models/` ignore rules cover all 21 paths; no ignore-rule or dataset-receipt change was required. No new DVC target was created for these pilot artifacts.

All 20 pilot features are tensors with shape `(16, 1024)`. The checkpoint stores model state and training metadata for the legacy pilot. No specific permission to redistribute these dataset-derived artifacts was documented. Their exclusion is precautionary and does not alter architecture, experiment definitions, results or local research assets. Historical removal remains separately approval-gated.

The following exact paths were found in historical Git changes, with no additional pilot tensor/checkpoint path names discovered in the locally available history. Their original SHA-256 values are the verification baseline:

| Historical and local path | Original SHA-256 |
|---|---|
| `temporal/features/fake/183_253.pt` | `f572b3cfb31a1eacb84785a82b58516e2429a5481007122979bcc7341bcc052e` |
| `temporal/features/fake/253_183.pt` | `7ccd5bd04daa4367dc9644496441db5c1840de21aabce99f8906a7a07ed885a8` |
| `temporal/features/fake/469_481.pt` | `d9c9b07aae2c98cd4ca420d32ff64d1e32a49eb3372a3240337260ec08ad157a` |
| `temporal/features/fake/481_469.pt` | `79e71c02871bd6a3f8be0315fccad5f4ff706381899635a9a24e46a8e5fa2b32` |
| `temporal/features/fake/585_599.pt` | `900545e2209cee120001cf4a5562c5efdc6113c21a504f02a278756780e3dd24` |
| `temporal/features/fake/599_585.pt` | `14b26ce58df36edc42a8d6b54f526bb5111d846403a4945549f42dc7c794572b` |
| `temporal/features/fake/672_720.pt` | `160aa56a6a6ad9b7a3974ef18acfe3b0ed92797f3aab7b01caf79e7c5ff492a7` |
| `temporal/features/fake/720_672.pt` | `3d64fdae8fafb08b46dfa114c79d749375caec210da143cc77c44f742f7b51e1` |
| `temporal/features/fake/866_878.pt` | `813ed781d27db0895bc0c4f8c48434cac7e8469889f1ff8b5df5f801f680922e` |
| `temporal/features/fake/878_866.pt` | `03cc18e918def3229a8d0a2f865c3a1152a67e2897f7768df785409548a526ae` |
| `temporal/features/real/183.pt` | `9e9d4ebb61d168caf5faab8e8ca8cde7967ad6df50ca5c02a2042bfd437cd199` |
| `temporal/features/real/253.pt` | `4bd42c1cc91be052f928feba8989bf43ae249b1e5cb9741caaf7a10a08da5e8a` |
| `temporal/features/real/469.pt` | `91b297f08913795ec5554b5c63e1220ee63bc92b927eef8b4dcf649b824f1e1f` |
| `temporal/features/real/481.pt` | `068af5de771376e8580ea18bdc73dcf8fac4c4e5170713c839557758ae5f1181` |
| `temporal/features/real/585.pt` | `6b6f4c12b0acfccd29f6a5c8fc058369c58e7b9a8cfb3c40d4b34656b950def1` |
| `temporal/features/real/599.pt` | `d9626509075e0d374d85a040255bb9f9d6f4f3b81e2caed8ca144610e291eac0` |
| `temporal/features/real/672.pt` | `cb531e36fd676c80ff3cca714a31cc31a7047669c30ef5b2b4affa6b8b4c2a10` |
| `temporal/features/real/720.pt` | `9dc236c15f5b33a55c92dd4d3f4955ff8dd0d6df671a4225613d18ae2944e04e` |
| `temporal/features/real/866.pt` | `ea4165eff88dce7b981a016061b8d5a2e36197ec5347997b514d35ac2d491825` |
| `temporal/features/real/878.pt` | `789adbcfb002315aa05f861b6d70dc79346fff4fcb2992066829bc7fc3949519` |
| `temporal/models/best_gru.pt` | `3434ecf0d025f5c1598bd64bab52bfd623574ae4da47d4fba2ddb4bf357c0bf0` |

The final approved deletion scope is **1073 files: 1020 MP4 videos, 32 face images, 20 pilot feature tensors and one legacy checkpoint**. Post-untracking SHA-256 comparison confirms that every targeted file remains locally available and matches its original SHA-256. All 21 pilot paths pass `git check-ignore`; none of the 1073 targets remains in the Git index. Both manifest hashes and all 54 result files still match the original baseline. Temporal contract checks passed again; frontend tests retain the same baseline outcome (10 passed, 3 Gradio compatibility errors, 1 FFmpeg-related skip). The dataset DVC receipt is unchanged by this extension (SHA-256 `b187c34dd1b20d868d88e3bf1de4840d100b5676fd588df2fef3306eba2b3d54`).

The remote inventory was refreshed read-only: five branch heads, no advertised tags, and PR head refs #1?#5. PR #1 (`ceb77df3850ae44b19745b30dd9aaeceb9e21af9`) is unavailable locally and remains unverified; no PR content was fetched into the research checkout. PRs #2?#5 reference locally available historical commits containing restricted assets. Exact branch/PR object IDs and disposable-clone inspection instructions are recorded in the history plan. Server-managed refs, cached views and forks are outside the local verification boundary.
