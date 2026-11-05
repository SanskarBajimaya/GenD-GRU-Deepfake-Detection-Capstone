# Proposed Git history remediation — do not execute during initial cleanup

Status: planning only. No history has been rewritten or purged. The owner must separately review and approve the exact paths, reference inventory, preservation evidence, maintenance window, and force-push scope.

## Scope and prerequisites

Confirmed restricted paths across inspected history:

```text
data/faceforensic/
data/faceforensics/
datasets/FF/
datasets/CDFv2/
temporal/features/
temporal/models/best_gru.pt
```

The plural `data/faceforensics/` is a previous path containing 20 raw video names. Inspect every historical path change again before rewriting. The 20 pilot tensors under `temporal/features/` and legacy `temporal/models/best_gru.pt` are now approved for local-only storage and included in the proposed historical filter. This is approval for file-level untracking only; history rewriting still requires separate approval. Review historical `media/gradio.mp4` provenance separately. Do not broadly filter all images, archives, tensors, `data/`, `datasets/`, or `temporal/`: that could remove receipts, code, tokenizer vocabulary, approved reports or screenshots.

Preserve `src/`, `detector.py`, all temporal source, `frontend/`, both manifests, all 54 result files, graphs, screenshots and documentation. Record their Git blob IDs and independent SHA-256 values before rewriting and compare each applicable branch/tag tip afterward. Preserve approved metadata and reports on historical commits too; inspect the filter's commit map and representative rewritten trees.

Freeze updates and coordinate with every collaborator. Back up uncommitted work and authorized local research assets outside working checkouts before anyone pulls the deletion commit. The existing verified same-computer backup is temporary: obtain an independent, access-controlled repository/data backup and verify restoration before rewriting. Never put restricted history in public backup storage. Do not modify the original repository or its backup.

## Recorded remote references and PR inspection gate

Read-only remote inventory on 2026-11-05 (refresh before rewriting):

| Remote ref | Object ID |
|---|---|
| `refs/heads/Dataset-cleanup` | `499fdbbb9fef73bbf6fdb96c6807d470a7aaa42a` |
| `refs/heads/Read-me-updates` | `499fdbbb9fef73bbf6fdb96c6807d470a7aaa42a` |
| `refs/heads/feature/database-pipeline` | `8f3c739f53d0b077fd9c22bd5e57b926601899e7` |
| `refs/heads/gru-pe` | `7dd9ae8ddd43c8babee7f7a3655153a39b99f135` |
| `refs/heads/main` | `0b4f7451d1252bb39d2d0437b8f0951e51ae5340` |
| `refs/pull/1/head` | `ceb77df3850ae44b19745b30dd9aaeceb9e21af9` |
| `refs/pull/2/head` | `7dd9ae8ddd43c8babee7f7a3655153a39b99f135` |
| `refs/pull/3/head` | `13528bcdc5c20b9bf6eef6a232715e813bd14a70` |
| `refs/pull/4/head` | `499fdbbb9fef73bbf6fdb96c6807d470a7aaa42a` |
| `refs/pull/5/head` | `8f3c739f53d0b077fd9c22bd5e57b926601899e7` |

Five branch heads and no tags were advertised. No PR merge refs were advertised; this does not prove that inaccessible server references are absent. The local `main` and some remote-tracking refs are older than this remote inventory; do not use them as a substitute for current remote discovery. Local cleanup/Codex refs also need consideration in local object-reachability checks; do not publish these bookkeeping refs.

PR #1 is **unverified**: its commit is not available in the research checkout. PR #2?#5 heads are available locally and contain restricted assets. Do not claim all PR histories have been audited. Before rewriting, obtain approval for a private disposable mirror and sufficient capacity, then run the following only inside that mirror. These commands can download restricted historical objects; never run them in the research checkout or the verified backup, and never upload their output publicly.

```powershell
git fetch origin refs/pull/1/head:refs/review/pull/1/head
git rev-parse refs/review/pull/1/head
git ls-tree -r --name-only refs/review/pull/1/head
git log refs/review/pull/1/head --format= --name-status --find-renames
git rev-list --objects refs/review/pull/1/head
git log refs/review/pull/1/head --format= --name-only -- data/faceforensic/ data/faceforensics/ datasets/FF/ datasets/CDFv2/ temporal/features/ temporal/models/best_gru.pt
```

Compare the fetched SHA with the refreshed advertisement; stop if access is denied or the reference changes unexpectedly. Inspect names and object metadata without checking out, running project code or extracting videos. Review the complete PR history for earlier names and content aliases, add discovered restricted paths/object IDs to the approved scope, and save the findings privately. Explicitly fetch/review all other advertised PR heads or merge refs into `refs/review/` if the mirror did not include them. Ensure these local review refs are rewritten too and included in post-filter verification; never push them to GitHub. PR server refs require GitHub handling rather than ordinary branch updates. Reassess fork and cache exposure separately.

Exact known historical pilot paths (SHA-256 preservation evidence is in the cleanup audit):

```text
temporal/features/fake/183_253.pt
temporal/features/fake/253_183.pt
temporal/features/fake/469_481.pt
temporal/features/fake/481_469.pt
temporal/features/fake/585_599.pt
temporal/features/fake/599_585.pt
temporal/features/fake/672_720.pt
temporal/features/fake/720_672.pt
temporal/features/fake/866_878.pt
temporal/features/fake/878_866.pt
temporal/features/real/183.pt
temporal/features/real/253.pt
temporal/features/real/469.pt
temporal/features/real/481.pt
temporal/features/real/585.pt
temporal/features/real/599.pt
temporal/features/real/672.pt
temporal/features/real/720.pt
temporal/features/real/866.pt
temporal/features/real/878.pt
temporal/models/best_gru.pt
```

No alternate pilot path names were found in the currently available local history. PR #1 and inaccessible refs may reveal additional names; inspection is a prerequisite, not a completed check.

## Proposed commands: owner-run in a disposable private location only

These commands are a reviewable proposal, not an executable script for this task. Use a sufficiently recent `git-filter-repo` supporting `--sensitive-data-removal` (GitHub specifies version 2.47 or later). Install it separately; confirm capacity for original backups plus a full disposable mirror. A mirror clone downloads restricted historical objects: its location must be private and authorized.

```powershell
git ls-remote https://github.com/SanskarBajimaya/Capstone---GRU-and-GenD-Deepfake-Detector-.git 'refs/heads/*' 'refs/tags/*' 'refs/pull/*'
git clone --mirror https://github.com/SanskarBajimaya/Capstone---GRU-and-GenD-Deepfake-Detector-.git capstone-history-review.git
Set-Location capstone-history-review.git
git for-each-ref --format='%(refname) %(objectname)'
git log --all --format= --name-status --find-renames
git rev-list --objects --all
git fsck --full
git filter-repo --version
```

Save the reference inventory, original restricted object IDs, preservation hashes, and review notes privately **outside the disposable clone**. Include all branches, tags, any fetched PR refs and other applicable refs; the initial local audit is not exhaustive of GitHub internals. Do not delete branches or tags to hide exposure. The following filter runs only after a separate owner approval:

```powershell
git filter-repo --sensitive-data-removal --invert-paths --path data/faceforensic/ --path data/faceforensics/ --path datasets/FF/ --path datasets/CDFv2/ --path temporal/features/ --path temporal/models/best_gru.pt
```

## Verify before considering any publication

```powershell
git for-each-ref --format='%(refname) %(objectname)'
git log --all --format= --name-only -- data/faceforensic/ data/faceforensics/ datasets/FF/ datasets/CDFv2/ temporal/features/ temporal/models/best_gru.pt
git rev-list --objects --all
git fsck --full
```

The targeted path log must be empty. Compare all old restricted blob IDs against the new reachable object-ID set, independent of path names, and investigate every surviving match. For each targeted object, `git cat-file -e <object-id>` should fail in the disposable rewritten repository if it has truly been removed from that object database; reachability checks alone cannot prove physical absence. Check unreachable objects with `git fsck --full --unreachable`. Stop on retained target content rather than running garbage collection against the research checkout. Review shared blobs individually if a target also appears under another filename.

Compare all approved source/manifests/results/asset blob IDs and SHA-256 values against the saved baseline. Run lightweight temporal and frontend tests in a separate checkout of rewritten tips; do not restore raw data into Git. Retain the filter report/commit map privately. Repeat remote ref discovery immediately before publication; reconcile newly created refs rather than overwriting teammates' work.

## Publication gate and coordination

Rewriting changes commit hashes and may invalidate signatures, PR review anchors, CI references, release references and open branches. Coordinate open PRs and branch protections in advance. Old clones, backup refs, unrebased branches and forks can reintroduce the data; require fresh clones and carefully reviewed patches for unmerged work, with research data separately backed up. Do not merge old histories into the rewritten repository.

`git-filter-repo` may remove the origin remote. After successful verification and **another explicit publication approval**, the owner can restore it and push individually reviewed branch/tag refs. Example for main only (replace the placeholder with the recorded, still-current remote SHA):

```powershell
git remote add origin https://github.com/SanskarBajimaya/Capstone---GRU-and-GenD-Deepfake-Detector-.git
git push --force-with-lease=refs/heads/main:REVIEWED_OLD_MAIN_SHA origin refs/heads/main:refs/heads/main
```

Use `git remote set-url` instead if origin remains. Repeat explicit leased updates for every approved affected branch/tag; do not issue a blanket mirror push, which may delete unrelated refs. No push commands are authorized by the initial cleanup task.

GitHub PR refs are generally server-managed and cannot be repaired through ordinary branch pushes. Old PR views, cached SHA URLs, forks and clones may retain content. Ask GitHub Support to assess affected PR refs/cached views and server retention using the filter report and affected refs, without attaching dataset content. Support's sensitive-data removal policy may not cover licensing-only data; complete removal is not guaranteed. Coordinate fork owners separately. A local rewrite or successful push cannot prove deletion from external copies or GitHub caches.

References: [GitHub removal guidance](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository), [git-filter-repo manual](https://github.com/newren/git-filter-repo/blob/main/Documentation/git-filter-repo.txt). These are procedural references, not a claim that this repository has been purged.
