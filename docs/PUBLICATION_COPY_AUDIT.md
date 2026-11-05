# Publication copy audit

Status: local preparation only; not cleared for upload. No Git repository was initialized, and no commit, upload, history rewrite or DVC operation was performed.

## Source and extraction

- Source commit: `896055860fac3cf2b0ebca929ebbaa060d9cf7ef`.
- Reference ZIP: `capstone-8960558-reference.zip (stored outside this repository)`.
- Reference ZIP SHA-256: `559df0bf333127c050ce9791860c526c968d49d083f7ef0b160c714954900e14`.
- Publication directory: `publication working directory (local location omitted)`.
- All 188 reference files were extracted and verified byte-for-byte before changes.
- The reference ZIP uses committed Git blob bytes (LF for text); existing Windows working-copy hashes may differ because of CRLF.
- Final expected file count: **188** (188 reference files minus one removed image plus this audit).

## Authorized changes

- Removed only publication-copy `media/gradio.png`, then removed its empty `media/` directory. No other media files were present.
- Modified exactly ten CSV checkpoint_path cells, listed below. All columns, row order, metric strings, labels, seeds, epochs and every other cell match the reference. No result was recalculated.
- README required no changes: both frontend screenshots and all five graphs are already linked, and no deleted media asset is linked.
- Historical media mentions in the old cleanup audit and history plan remain historical records, not live image links. No frontend runtime dependency on media/ was found.
- Two commented upstream /media/... example paths in src/model/forada/config.yaml remain unchanged because source preservation is required.

## Exact checkpoint path replacements

The full original path-replacement evidence is preserved in a separate access-restricted local audit outside this repository. Personal absolute prefixes are redacted here; checkpoint filenames and repository-relative replacements are retained. The ten corrected CSVs are unchanged by this redaction.

| CSV | Original checkpoint_path | Replacement |
|---|---|---|
| `temporal/results/gru32_seed0_run.csv` | `[REDACTED_PRIVATE_ROOT]/temporal/models/gru32_seed0_best.pt` | `temporal/models/gru32_seed0_best.pt` |
| `temporal/results/gru32_seed1_run.csv` | `[REDACTED_PRIVATE_ROOT]/temporal/models/gru32_seed1_best.pt` | `temporal/models/gru32_seed1_best.pt` |
| `temporal/results/gru32_seed2_run.csv` | `[REDACTED_PRIVATE_ROOT]/temporal/models/gru32_seed2_best.pt` | `temporal/models/gru32_seed2_best.pt` |
| `temporal/results/gru32_seed3_run.csv` | `[REDACTED_PRIVATE_ROOT]/temporal/models/gru32_seed3_best.pt` | `temporal/models/gru32_seed3_best.pt` |
| `temporal/results/gru32_seed4_run.csv` | `[REDACTED_PRIVATE_ROOT]/temporal/models/gru32_seed4_best.pt` | `temporal/models/gru32_seed4_best.pt` |
| `temporal/results/gru64_seed0_run.csv` | `[REDACTED_PRIVATE_ROOT]/temporal/models/gru64_seed0_best.pt` | `temporal/models/gru64_seed0_best.pt` |
| `temporal/results/gru64_seed1_run.csv` | `[REDACTED_PRIVATE_ROOT]/temporal/models/gru64_seed1_best.pt` | `temporal/models/gru64_seed1_best.pt` |
| `temporal/results/gru64_seed2_run.csv` | `[REDACTED_PRIVATE_ROOT]/temporal/models/gru64_seed2_best.pt` | `temporal/models/gru64_seed2_best.pt` |
| `temporal/results/gru64_seed3_run.csv` | `[REDACTED_PRIVATE_ROOT]/temporal/models/gru64_seed3_best.pt` | `temporal/models/gru64_seed3_best.pt` |
| `temporal/results/gru64_seed4_run.csv` | `[REDACTED_PRIVATE_ROOT]/temporal/models/gru64_seed4_best.pt` | `temporal/models/gru64_seed4_best.pt` |

## Preservation and exclusions

- Both frontend screenshots are byte-for-byte unchanged: docs/readme-assets/frontend-real.jpg.jpeg and docs/readme-assets/frontend-fake.jpg.jpeg. Inclusion is approved for preparation only; redistribution permissions remain unverified.
- All five numerical result graphs are byte-for-byte unchanged.
- Both manifests are byte-for-byte identical to the reference ZIP.
- All 54 result files are present. Forty-four are byte-for-byte unchanged; ten differ only in their approved checkpoint_path field.
- All source, temporal backend, detector.py, frontend, LICENSE, attribution, README, DVC configuration and data/faceforensic.dvc are unchanged.
- No old .git directory/history, raw videos, dataset frame directories, pilot tensors, legacy checkpoint, trained checkpoints, DVC cache, credentials or machine-local DVC config were extracted.
- The original checkout, research data, backup and reference ZIP were not modified.

## Remaining publication concerns

- The two retained screenshots contain recognizable video frames. Obtain the applicable redistribution permission before uploading; preparation approval is not publication approval.
- Personal absolute paths have been redacted from this publication audit. The unredacted evidence is retained privately outside the repository.
- Protected upstream source has commented absolute example paths; review separately before release rather than editing source without approval.
- Automated credential-pattern screening is not a comprehensive security or license clearance.
- Legacy history-remediation documents describe the original repository. A future independent repository must start with fresh Git history and must not import old refs or merges.
- Dataset files, trained GRU checkpoints and native/runtime dependencies remain separately required for relevant workflows; this copy does not invent missing assets.

## File-by-file comparison against reference ZIP

SHA-256 values below describe the reference and the prepared files. The audit is the sole newly added file and cannot include its own final hash without self-reference.

| Path | Status | Reference SHA-256 | Prepared SHA-256 |
|---|---|---|
| `.dvc/.gitignore` | unchanged | `cda3fc1a4ce9659cb19818bb065eee8c5271795484f49f314ca0357b4daec6ef` | `cda3fc1a4ce9659cb19818bb065eee8c5271795484f49f314ca0357b4daec6ef` |
| `.dvc/config` | unchanged | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `.dvcignore` | unchanged | `1d8052e9672c44c62dc27a4df73ced356f20cb9c18d1a7b5aa09249309609583` | `1d8052e9672c44c62dc27a4df73ced356f20cb9c18d1a7b5aa09249309609583` |
| `.gitignore` | unchanged | `77c11fc2eb7b37921ce8be3baf8a41efc8575624e9d2322ef55674cbd6d7b786` | `77c11fc2eb7b37921ce8be3baf8a41efc8575624e9d2322ef55674cbd6d7b786` |
| `.project-root` | unchanged | `ee158bbff1d6cea9c14314a1c4d181698dcd801c2eda352f37454a22d59ed77a` | `ee158bbff1d6cea9c14314a1c4d181698dcd801c2eda352f37454a22d59ed77a` |
| `LICENSE` | unchanged | `f478adb5d8cbcb41e0236408a98a0b44269798b026e73f2da16b9cc0e55f842a` | `f478adb5d8cbcb41e0236408a98a0b44269798b026e73f2da16b9cc0e55f842a` |
| `README.md` | unchanged | `2eb48b8b3091e2744ac62d98c9dadd60e1df93185582806974c39128588e4033` | `2eb48b8b3091e2744ac62d98c9dadd60e1df93185582806974c39128588e4033` |
| `app/run.py` | unchanged | `97432055d9330f3cdc59bb97ee82579483306d84a7cc4506efdc8393db13f891` | `97432055d9330f3cdc59bb97ee82579483306d84a7cc4506efdc8393db13f891` |
| `config/datasets/CDFv2/test/Celeb-real.txt` | unchanged | `d83709aaca998d0f72cef0618394d7b679a04bcf8494fd6f28d7a85910f02a19` | `d83709aaca998d0f72cef0618394d7b679a04bcf8494fd6f28d7a85910f02a19` |
| `config/datasets/CDFv2/test/Celeb-synthesis.txt` | unchanged | `0873e6695cd1554629c187032f7894c5b73448dded6ac2809686ff09abb12d55` | `0873e6695cd1554629c187032f7894c5b73448dded6ac2809686ff09abb12d55` |
| `config/datasets/CDFv2/test/YouTube-real.txt` | unchanged | `a37e4469b2a39f7b494971423ac4c032046703fe88fe3161940f798c1fe3b5d5` | `a37e4469b2a39f7b494971423ac4c032046703fe88fe3161940f798c1fe3b5d5` |
| `config/datasets/FF/test/DF.txt` | unchanged | `ff5866b22e26a961cc6febf91a86f931911e0730909dba821ed3d9157e55551d` | `ff5866b22e26a961cc6febf91a86f931911e0730909dba821ed3d9157e55551d` |
| `config/datasets/FF/test/F2F.txt` | unchanged | `967dc1187c404b4c22d8f6e77f1f1ea8b9fed19a41fef0fd56371f6e4cdfd287` | `967dc1187c404b4c22d8f6e77f1f1ea8b9fed19a41fef0fd56371f6e4cdfd287` |
| `config/datasets/FF/test/FS.txt` | unchanged | `f91c20392c6e199015fd5722bf968982591e7d488d8b13c06fb93803c9ad5cfb` | `f91c20392c6e199015fd5722bf968982591e7d488d8b13c06fb93803c9ad5cfb` |
| `config/datasets/FF/test/NT.txt` | unchanged | `e7c9cbd8717085a6191c55b783b493b8cfc8f1c9ce923f6ba5d395fe714d18b8` | `e7c9cbd8717085a6191c55b783b493b8cfc8f1c9ce923f6ba5d395fe714d18b8` |
| `config/datasets/FF/test/real.txt` | unchanged | `66a4e1666fd828ea2db0a45ef7a5fa300266d755150e759ce36da5671d7439fa` | `66a4e1666fd828ea2db0a45ef7a5fa300266d755150e759ce36da5671d7439fa` |
| `data/README.md` | unchanged | `1f98fae721356554c65710b361398bf42a4a6da3b2ea9c7b6ce37e078f225fb9` | `1f98fae721356554c65710b361398bf42a4a6da3b2ea9c7b6ce37e078f225fb9` |
| `data/faceforensic.dvc` | unchanged | `d1bd262b9f85d4eea8ffa5d915a32baa6f69a97609645f514d0aaacb2d50e726` | `d1bd262b9f85d4eea8ffa5d915a32baa6f69a97609645f514d0aaacb2d50e726` |
| `detector.py` | unchanged | `c7cbdf2dc37ffd5612f3a6b5583027e8eaa42e5eb261eae447bf92b1919eb7b7` | `c7cbdf2dc37ffd5612f3a6b5583027e8eaa42e5eb261eae447bf92b1919eb7b7` |
| `docs/DATASET_CLEANUP_AUDIT.md` | unchanged | `9c48e5db1e811a3c8ec6eeff9da604cf7d7a0df6b01451e91407cac3f1412033` | `9c48e5db1e811a3c8ec6eeff9da604cf7d7a0df6b01451e91407cac3f1412033` |
| `docs/GIT_HISTORY_CLEANUP_PLAN.md` | unchanged | `866c751eeaabf5631b7a3d1256d013b3e59c48ae4ab9b45a41c28b6b9d344941` | `866c751eeaabf5631b7a3d1256d013b3e59c48ae4ab9b45a41c28b6b9d344941` |
| `docs/LOCAL_DVC_SETUP.md` | unchanged | `3b13e633769adbae7ebf48814aa6a21ce82c6de8eb821ed9fd9d48d1eea696cc` | `3b13e633769adbae7ebf48814aa6a21ce82c6de8eb821ed9fd9d48d1eea696cc` |
| `docs/generate_readme_graphs.py` | unchanged | `76fdf5fe7b1dddc454b157573f691b785a10d232bcda27deb4eba867a2776c30` | `76fdf5fe7b1dddc454b157573f691b785a10d232bcda27deb4eba867a2776c30` |
| `docs/readme-assets/celebdf-test-auroc.png` | unchanged | `15a459c3b2446cec9923f26124f1207d18ef5f482fedec3ca77065da2ec4dc40` | `15a459c3b2446cec9923f26124f1207d18ef5f482fedec3ca77065da2ec4dc40` |
| `docs/readme-assets/ffpp-test-auroc.png` | unchanged | `d4a74763eb5e802b5c2266f93654df3ac5bc2d821c04f27ff7b168dbe0c4da95` | `d4a74763eb5e802b5c2266f93654df3ac5bc2d821c04f27ff7b168dbe0c4da95` |
| `docs/readme-assets/frontend-fake.jpg.jpeg` | unchanged | `53b84551e48d6262832c18b5996e3aa18ad51321b586a1d5b5b406ff233e63b3` | `53b84551e48d6262832c18b5996e3aa18ad51321b586a1d5b5b406ff233e63b3` |
| `docs/readme-assets/frontend-real.jpg.jpeg` | unchanged | `e5e774dd74077062f26911d9bfbecc85900e9e26420e9b43395fff33113bd217` | `e5e774dd74077062f26911d9bfbecc85900e9e26420e9b43395fff33113bd217` |
| `docs/readme-assets/train-bce.png` | unchanged | `4e9c274de9369b25c2ba30e864eba4ebbe63b830f3f217fdfab4adc2074e192f` | `4e9c274de9369b25c2ba30e864eba4ebbe63b830f3f217fdfab4adc2074e192f` |
| `docs/readme-assets/validation-auroc.png` | unchanged | `d4a4903150a1725888de24de52c1e4caa212808f2a8efdacf2d3f99fa37bc4f4` | `d4a4903150a1725888de24de52c1e4caa212808f2a8efdacf2d3f99fa37bc4f4` |
| `docs/readme-assets/validation-bce.png` | unchanged | `ebfb5c61e46b3f2bc73c6ced2af7d1f967f602fc790ddf1988e5180fbcb17284` | `ebfb5c61e46b3f2bc73c6ced2af7d1f967f602fc790ddf1988e5180fbcb17284` |
| `frontend/.gitignore` | unchanged | `5ad3d2f198995b3f80c2d0101d95d3d9822cb1010e980bae5b32a7dbca39bad5` | `5ad3d2f198995b3f80c2d0101d95d3d9822cb1010e980bae5b32a7dbca39bad5` |
| `frontend/README.md` | unchanged | `9be2b1b59c2a79d844c977aa069019698c9c85e1adad0e5543675873de0c3e6b` | `9be2b1b59c2a79d844c977aa069019698c9c85e1adad0e5543675873de0c3e6b` |
| `frontend/app.py` | unchanged | `209d12499c173d2a50627563c423f2d063d162692592a0784628a8aeff12933b` | `209d12499c173d2a50627563c423f2d063d162692592a0784628a8aeff12933b` |
| `frontend/model_loader.py` | unchanged | `60e3ce6c1c00dd6cab82d0e0e3a9f811a65b77616b724170cec1a1455d25bf4b` | `60e3ce6c1c00dd6cab82d0e0e3a9f811a65b77616b724170cec1a1455d25bf4b` |
| `frontend/predictor.py` | unchanged | `a8ca8fcf037c2c37e9e1f24a6145dc076eb002b47f94d166948fe29b9d0ced65` | `a8ca8fcf037c2c37e9e1f24a6145dc076eb002b47f94d166948fe29b9d0ced65` |
| `frontend/preprocessing.py` | unchanged | `4c6563b72525cf77111f65889e55b37470b18d9834f388cc2df7934913a2b8f9` | `4c6563b72525cf77111f65889e55b37470b18d9834f388cc2df7934913a2b8f9` |
| `frontend/repository.py` | unchanged | `af2691ef557388319459b38c4654bcea00ef7a45e28d39ee39e8027d656ae583` | `af2691ef557388319459b38c4654bcea00ef7a45e28d39ee39e8027d656ae583` |
| `frontend/requirements.txt` | unchanged | `8979ef348ac898e047f972098838fa2adad5c1ca61e2c23c39f8fb5f8eda5776` | `8979ef348ac898e047f972098838fa2adad5c1ca61e2c23c39f8fb5f8eda5776` |
| `frontend/style.css` | unchanged | `ae49a9f272eb7369161adf9f0cd40e61d5058477bc9ddbbef1c03e3e3957f5a8` | `ae49a9f272eb7369161adf9f0cd40e61d5058477bc9ddbbef1c03e3e3957f5a8` |
| `frontend/tests/__init__.py` | unchanged | `196ecaaf65474ddf14aedeb6ebfa2e7ce4e46a4d4829f5ec005900bb8ec72199` | `196ecaaf65474ddf14aedeb6ebfa2e7ce4e46a4d4829f5ec005900bb8ec72199` |
| `frontend/tests/test_frontend.py` | unchanged | `5d7b58b559ebb1590dc970235a51ab3ab032b96f41c74a31417f34ce29723089` | `5d7b58b559ebb1590dc970235a51ab3ab032b96f41c74a31417f34ce29723089` |
| `media/gradio.png` | removed | `4c1513b26c43f38d97790d55a8b7b74a846f0c5f2847bdfec550a8da27ad6dbb` | `absent` |
| `pyproject.toml` | unchanged | `129827e9982654eaa53954fefecce77bf9a89cb5d2159efb8d0f4e3310c8b7ff` | `129827e9982654eaa53954fefecce77bf9a89cb5d2159efb8d0f4e3310c8b7ff` |
| `requirements.txt` | unchanged | `0284748a9329b3e947da62af67ec660b291f44a56a78e65c4f1d7ee672e4f213` | `0284748a9329b3e947da62af67ec660b291f44a56a78e65c4f1d7ee672e4f213` |
| `run.py` | unchanged | `cc09b604c77ccc717eb9eabf1be9c4dfe3404b0be2ff6bbacf025c2ba69de516` | `cc09b604c77ccc717eb9eabf1be9c4dfe3404b0be2ff6bbacf025c2ba69de516` |
| `run_exp.py` | unchanged | `822d74a3cfd66a48b427fd5a4b9b4e3d52ac82c958ea01e5b93d7628318fe39e` | `822d74a3cfd66a48b427fd5a4b9b4e3d52ac82c958ea01e5b93d7628318fe39e` |
| `src/__init__.py` | unchanged | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/config.py` | unchanged | `51d034d8aa50669d7f30f211a50405d8c3a33ede4b19a0e3a3eda205b0b65338` | `51d034d8aa50669d7f30f211a50405d8c3a33ede4b19a0e3a3eda205b0b65338` |
| `src/dataset/__init__.py` | unchanged | `4265cc1a7fdf8088ae39bc86a8eed9ea8544c06b27b1a4263ecbda5a5b4c0c9f` | `4265cc1a7fdf8088ae39bc86a8eed9ea8544c06b27b1a4263ecbda5a5b4c0c9f` |
| `src/dataset/augmentations.py` | unchanged | `c369d98d8e02f35ec4efdb0590fa461390453e441810745bfa285a114b4b4171` | `c369d98d8e02f35ec4efdb0590fa461390453e441810745bfa285a114b4b4171` |
| `src/dataset/base.py` | unchanged | `8998122865e325e13d570730057223010436775cba87ba90430244a9edf0287f` | `8998122865e325e13d570730057223010436775cba87ba90430244a9edf0287f` |
| `src/dataset/data_module.py` | unchanged | `f7c267051e65538da6938cdace9b75cbd2ad4d464515463a31789a9abb35d413` | `f7c267051e65538da6938cdace9b75cbd2ad4d464515463a31789a9abb35d413` |
| `src/dataset/dataset.py` | unchanged | `cb118c77fcf239875c56c53a18c7d93d909969373747180c2ce502910ef45d5e` | `cb118c77fcf239875c56c53a18c7d93d909969373747180c2ce502910ef45d5e` |
| `src/encoders/_common.py` | unchanged | `86eef3e7773bc77c9b04ad8da2b7b46ad04553e5d2ac0013a6aa98f9cf9a25b2` | `86eef3e7773bc77c9b04ad8da2b7b46ad04553e5d2ac0013a6aa98f9cf9a25b2` |
| `src/encoders/clip_encoder.py` | unchanged | `beb5d51157b028f0dee9a68b7ee7675a4838c261814e122f9f33341ec5d0803c` | `beb5d51157b028f0dee9a68b7ee7675a4838c261814e122f9f33341ec5d0803c` |
| `src/encoders/dino_encoder.py` | unchanged | `1981c20b1308f460c1cdde8b0848ae744c87d27cbb3d5e4aaa1cbf23902b2d09` | `1981c20b1308f460c1cdde8b0848ae744c87d27cbb3d5e4aaa1cbf23902b2d09` |
| `src/encoders/perception_encoder.py` | unchanged | `8af6cc454973078b84cc069a307bdd0b1113b7e590414beefbd7abda52d7ade4` | `8af6cc454973078b84cc069a307bdd0b1113b7e590414beefbd7abda52d7ade4` |
| `src/exp/__init__.py` | unchanged | `6ae443d08f2228b1129ea0a46a097f90194bb0018d389481de065f55978d9658` | `6ae443d08f2228b1129ea0a46a097f90194bb0018d389481de065f55978d9658` |
| `src/exp/examples.py` | unchanged | `cdda9d2df6edf7d23ad21095b0afa86ad13c09b489cfeb4e7202ad8aefa1c26c` | `cdda9d2df6edf7d23ad21095b0afa86ad13c09b489cfeb4e7202ad8aefa1c26c` |
| `src/exp/third_party.py` | unchanged | `bb5e20a0b8470255b48919f15f3133c91b56b228e1b9b1fbb965c3e296cfeea9` | `bb5e20a0b8470255b48919f15f3133c91b56b228e1b9b1fbb965c3e296cfeea9` |
| `src/exp/wacv.sh` | unchanged | `f08003a02ffa6f2b52246a38405e7dc116caf070eafc288467971c26ea129ad4` | `f08003a02ffa6f2b52246a38405e7dc116caf070eafc288467971c26ea129ad4` |
| `src/exp/wacv_rebuttal.py` | unchanged | `66aa73ca72606c52b5b882bb6f49e0ed5b79a057cddd18aafe1f3701fd2e259f` | `66aa73ca72606c52b5b882bb6f49e0ed5b79a057cddd18aafe1f3701fd2e259f` |
| `src/exp/wacv_rebuttal_aug_robustness.py` | unchanged | `dba9ede88d3e8a9368befd7092c53df6f72d90a2bd76b8029ec0250082355fb3` | `dba9ede88d3e8a9368befd7092c53df6f72d90a2bd76b8029ec0250082355fb3` |
| `src/exp/wacv_rebuttal_paired_unpaired.py` | unchanged | `f427f8805dc360fe54d741fa766f6eb199c028b84f83a967cff990555b2399d2` | `f427f8805dc360fe54d741fa766f6eb199c028b84f83a967cff990555b2399d2` |
| `src/heads/head.py` | unchanged | `264fabb1f1da855882200bc132ed11249c9ce9a382ee7e729857ca57996d5a5c` | `264fabb1f1da855882200bc132ed11249c9ce9a382ee7e729857ca57996d5a5c` |
| `src/hf/modeling_gend.py` | unchanged | `4b76b83df8d0f5c59082e97ececeb67735eb79a1fa0fc09d9299710179b75ad0` | `4b76b83df8d0f5c59082e97ececeb67735eb79a1fa0fc09d9299710179b75ad0` |
| `src/loss.py` | unchanged | `e913a99a37842ba6ee975bc97932dfd2b3795d849d555ab5a0b9819998a212cc` | `e913a99a37842ba6ee975bc97932dfd2b3795d849d555ab5a0b9819998a212cc` |
| `src/losses/unifalign.py` | unchanged | `1732270c35faef87b98dc6d27eb935f92bd22e2861ed7542afbe4f668fa74521` | `1732270c35faef87b98dc6d27eb935f92bd22e2861ed7542afbe4f668fa74521` |
| `src/metrics.py` | unchanged | `b0d7e435822eff34c0a7fde9e31b1559dc30ccd4081a32bfd284e17cb731eb7a` | `b0d7e435822eff34c0a7fde9e31b1559dc30ccd4081a32bfd284e17cb731eb7a` |
| `src/model/Effort.py` | unchanged | `8448c7863f5270278f31647f10c6f28731fe4774d40ac474c27366ff576e25e6` | `8448c7863f5270278f31647f10c6f28731fe4774d40ac474c27366ff576e25e6` |
| `src/model/FSFM.py` | unchanged | `a8be21bfd712409577655eae169ad43e8ac7a4f2605a7a123bceb564579bc1a7` | `a8be21bfd712409577655eae169ad43e8ac7a4f2605a7a123bceb564579bc1a7` |
| `src/model/ForAda.py` | unchanged | `c399d7981349b9dcd3fa867ee9d84e7845951723a0df55043b547460db743508` | `c399d7981349b9dcd3fa867ee9d84e7845951723a0df55043b547460db743508` |
| `src/model/GenD.py` | unchanged | `b18179e7718ababf2cb4101d63926ee0600729cdc266c177d716d039fc5c639f` | `b18179e7718ababf2cb4101d63926ee0600729cdc266c177d716d039fc5c639f` |
| `src/model/GenDHF.py` | unchanged | `cb194e314eada34e2b4280cff7446e780e2306c46d68de2a448d6f2f39ae6f47` | `cb194e314eada34e2b4280cff7446e780e2306c46d68de2a448d6f2f39ae6f47` |
| `src/model/__init__.py` | unchanged | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/model/base.py` | unchanged | `11f474d844bc141650f00310b32fb924216ffda3ab2865da2a960c183fd3ecf5` | `11f474d844bc141650f00310b32fb924216ffda3ab2865da2a960c183fd3ecf5` |
| `src/model/effort/model.py` | unchanged | `f5c7676947b6419cfc9317821c3c93fd19a167209f8f224a371bd25240495c3d` | `f5c7676947b6419cfc9317821c3c93fd19a167209f8f224a371bd25240495c3d` |
| `src/model/forada/__init__.py` | unchanged | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/model/forada/adapters/adapter.py` | unchanged | `b0e79863551ef2766ee34c1d3632c14735931bb727f2281d8a97416ac6b598de` | `b0e79863551ef2766ee34c1d3632c14735931bb727f2281d8a97416ac6b598de` |
| `src/model/forada/attn.py` | unchanged | `1d8a26c9e5eca710b7df8762884eae11a91448c2d43be8544ac93d570888308c` | `1d8a26c9e5eca710b7df8762884eae11a91448c2d43be8544ac93d570888308c` |
| `src/model/forada/clip/__init__.py` | unchanged | `d7930eb290ebed432116795c91473200d098451a94c0aedd4ae0f934f97955e8` | `d7930eb290ebed432116795c91473200d098451a94c0aedd4ae0f934f97955e8` |
| `src/model/forada/clip/bpe_simple_vocab_16e6.txt.gz` | unchanged | `924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a` | `924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a` |
| `src/model/forada/clip/clip.py` | unchanged | `fd8cee5e74491f4e44f69a34dd9b644e91bf46f31bddad63383574ec6adf6f85` | `fd8cee5e74491f4e44f69a34dd9b644e91bf46f31bddad63383574ec6adf6f85` |
| `src/model/forada/clip/model.py` | unchanged | `4922235d97c3a45588991bd5f7d2ceed2e3fcf1b71010e56cef4acff6a138f8c` | `4922235d97c3a45588991bd5f7d2ceed2e3fcf1b71010e56cef4acff6a138f8c` |
| `src/model/forada/clip/simple_tokenizer.py` | unchanged | `dfdcb47fbd00f18a70010d4b4735ae89df12661b5c6be7d7effcab88a50e7d1a` | `dfdcb47fbd00f18a70010d4b4735ae89df12661b5c6be7d7effcab88a50e7d1a` |
| `src/model/forada/config.yaml` | unchanged | `b58d003bbf7698cf02908ce0b133a2c03ecf20fb53b396920b263c9e8618448e` | `b58d003bbf7698cf02908ce0b133a2c03ecf20fb53b396920b263c9e8618448e` |
| `src/model/forada/ds.py` | unchanged | `78c9bc7119100228d510b43fa94ce62464e25aafcec54ea85716032fbe1475b2` | `78c9bc7119100228d510b43fa94ce62464e25aafcec54ea85716032fbe1475b2` |
| `src/model/forada/layer.py` | unchanged | `ff13acd65ed4f34a9371544c3a3b9a3350b0581024703a2af78f6e8898f8a593` | `ff13acd65ed4f34a9371544c3a3b9a3350b0581024703a2af78f6e8898f8a593` |
| `src/model/fsfm/models_vit.py` | unchanged | `c3099660136010bb3339573893004e57fb593a5299fedab2b34898b3b1a36473` | `c3099660136010bb3339573893004e57fb593a5299fedab2b34898b3b1a36473` |
| `src/model/fsfm/models_vit_fs_adapter.py` | unchanged | `9f7c2c8b58477e89a68c3640901d17a470b20bbcce7948f8ccc023d878664d90` | `9f7c2c8b58477e89a68c3640901d17a470b20bbcce7948f8ccc023d878664d90` |
| `src/plots.py` | unchanged | `f97ddba70d55c969ffe17a495ddd8984d7037210ad9a40418946b38c16e31b12` | `f97ddba70d55c969ffe17a495ddd8984d7037210ad9a40418946b38c16e31b12` |
| `src/retinaface.py` | unchanged | `17cc9cacd7b166126412fa88ffb82002918821028a76a0a887e442cda061a99f` | `17cc9cacd7b166126412fa88ffb82002918821028a76a0a887e442cda061a99f` |
| `src/utils/checks.py` | unchanged | `a370d434b89c23c5fe72424dd16f52ffe52c82a0d784d2dfb2d6b7530bf67512` | `a370d434b89c23c5fe72424dd16f52ffe52c82a0d784d2dfb2d6b7530bf67512` |
| `src/utils/constants.py` | unchanged | `8e69c2ba33c8dc320a66f1a286b466280e2f51ae6143497750535bc60175e705` | `8e69c2ba33c8dc320a66f1a286b466280e2f51ae6143497750535bc60175e705` |
| `src/utils/decorators.py` | unchanged | `40a9b10fdf62e007584138f6ee1913d3b71491adeda2358a47bdd5875c72415c` | `40a9b10fdf62e007584138f6ee1913d3b71491adeda2358a47bdd5875c72415c` |
| `src/utils/files.py` | unchanged | `416268f77cb6e197c94715ff262925884234162c52d887fe73a6ad993205bc6a` | `416268f77cb6e197c94715ff262925884234162c52d887fe73a6ad993205bc6a` |
| `src/utils/logger.py` | unchanged | `b24167b36937ca80eeb26804a6b12751cabd3b042b6d681c17b113988394c620` | `b24167b36937ca80eeb26804a6b12751cabd3b042b6d681c17b113988394c620` |
| `src/utils/model_checkpoint.py` | unchanged | `10f2dd708178d8cfe629eb6d4cb3e0f7dfee7c5bd5d459b731483374578acad9` | `10f2dd708178d8cfe629eb6d4cb3e0f7dfee7c5bd5d459b731483374578acad9` |
| `src/utils/silencer.py` | unchanged | `93bcb872894c8a0061001b2f160900bc1a078fd118455ae89b1e6e02a055b989` | `93bcb872894c8a0061001b2f160900bc1a078fd118455ae89b1e6e02a055b989` |
| `src/utils/wb.py` | unchanged | `d6da120432375e32bcc5ab3945e49676676c18badeffa1ecbc3839713547c034` | `d6da120432375e32bcc5ab3945e49676676c18badeffa1ecbc3839713547c034` |
| `temporal/LARGE_EXPERIMENT.md` | unchanged | `a95ec1c3f4b6cc7d1e1a1fc0a0302acdbd975435d3c39f1050393165c8f334c3` | `a95ec1c3f4b6cc7d1e1a1fc0a0302acdbd975435d3c39f1050393165c8f334c3` |
| `temporal/__init__.py` | unchanged | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `temporal/build_feature_dataset.py` | unchanged | `24e6cabfeb8d4c63ea557a7d900d20e5f52d2cf4f9eeffe7fb1bb975474aa661` | `24e6cabfeb8d4c63ea557a7d900d20e5f52d2cf4f9eeffe7fb1bb975474aa661` |
| `temporal/build_manifest.py` | unchanged | `82070ff4211c1f8a50ddf94bf239c36b1f8ec71cb41dbe4fa185947e4ce40956` | `82070ff4211c1f8a50ddf94bf239c36b1f8ec71cb41dbe4fa185947e4ce40956` |
| `temporal/build_master_features.py` | unchanged | `7fc7b4dd205d89eb74e93a836c86b81ec672851015adc23c8110b78d4a96aee5` | `7fc7b4dd205d89eb74e93a836c86b81ec672851015adc23c8110b78d4a96aee5` |
| `temporal/build_pe_features.py` | unchanged | `1adecd590b250906589aad8b069a009168f14e6f6f7a2bd15011bd2b53df10c9` | `1adecd590b250906589aad8b069a009168f14e6f6f7a2bd15011bd2b53df10c9` |
| `temporal/check_gpu.py` | unchanged | `d6d2eab7bc5ed7a9fd60123f3cd329fa53253e2c6593f5eb948285a3f2edcc3c` | `d6d2eab7bc5ed7a9fd60123f3cd329fa53253e2c6593f5eb948285a3f2edcc3c` |
| `temporal/compare_all.py` | unchanged | `1a22f89f36af3ca8d6287bad2ee477814010a0a2776ed2960f74926293b2d534` | `1a22f89f36af3ca8d6287bad2ee477814010a0a2776ed2960f74926293b2d534` |
| `temporal/compare_gend_vs_gru.py` | unchanged | `9cd37c66a96c731cda9121367cc8490e6087e8866cbd812a8cc19cbdb0e9aac3` | `9cd37c66a96c731cda9121367cc8490e6087e8866cbd812a8cc19cbdb0e9aac3` |
| `temporal/compare_large_experiment.py` | unchanged | `fca246c2c67d739d4bf58fc4d12c0d095fd4622e614a8a67626f728b42f5f31d` | `fca246c2c67d739d4bf58fc4d12c0d095fd4622e614a8a67626f728b42f5f31d` |
| `temporal/dataset_split.py` | unchanged | `08709b955bf373ab498cf13284e0094ad1eca4ae4fc643cdc79e04f48069e49c` | `08709b955bf373ab498cf13284e0094ad1eca4ae4fc643cdc79e04f48069e49c` |
| `temporal/eval_celebdf_gru.py` | unchanged | `a008c56b5d8bf214578e10ebbb4914edf116f255c553b86856ce3964a8b4cffe` | `a008c56b5d8bf214578e10ebbb4914edf116f255c553b86856ce3964a8b4cffe` |
| `temporal/experiment_config.py` | unchanged | `63f1061ac967ae31b4ce49a5c24c17740a64f8ee8315f055ac07e6c2593fdf7d` | `63f1061ac967ae31b4ce49a5c24c17740a64f8ee8315f055ac07e6c2593fdf7d` |
| `temporal/extract_features.py` | unchanged | `d3b11482e8cd5728ca11ebcbc1c377af8a0de75c8a48f4d9c94f45454dea1286` | `d3b11482e8cd5728ca11ebcbc1c377af8a0de75c8a48f4d9c94f45454dea1286` |
| `temporal/feature_dataset.py` | unchanged | `03fe331d7426e293e9254af53cbf6cbf33e03922b163a763488e9cb1298d7f77` | `03fe331d7426e293e9254af53cbf6cbf33e03922b163a763488e9cb1298d7f77` |
| `temporal/large_experiment.py` | unchanged | `79475ae77c77a9b2f030316954fbf4c89fbc182fcb3d6ce8518ab0cda6b8609f` | `79475ae77c77a9b2f030316954fbf4c89fbc182fcb3d6ce8518ab0cda6b8609f` |
| `temporal/manifests/split_seed_2026.json` | unchanged | `de4d46626e4810af182588ba49a516ed6f556bed220045caf6f8c57d79f7d00c` | `de4d46626e4810af182588ba49a516ed6f556bed220045caf6f8c57d79f7d00c` |
| `temporal/manifests/video_manifest.json` | unchanged | `9868d860b262e3aae4903a4b037812233a2158f1ba40c47b6ac54c5c8ec9564a` | `9868d860b262e3aae4903a4b037812233a2158f1ba40c47b6ac54c5c8ec9564a` |
| `temporal/master_dataset.py` | unchanged | `b4fe9fdf5f0da49f676cbc14576ad23ae9fe68ff47704ad2f0ff121601c041bf` | `b4fe9fdf5f0da49f676cbc14576ad23ae9fe68ff47704ad2f0ff121601c041bf` |
| `temporal/model_a.py` | unchanged | `0a041f0aded22c499ef3d85f1e2103ed0764e66c832726f3f2698f5dcb1a7fb7` | `0a041f0aded22c499ef3d85f1e2103ed0764e66c832726f3f2698f5dcb1a7fb7` |
| `temporal/model_a_pipeline.py` | unchanged | `537a75278f61b09f7085d3b5c49bff17a894d54bf30cdb5f19dba610cff1d540` | `537a75278f61b09f7085d3b5c49bff17a894d54bf30cdb5f19dba610cff1d540` |
| `temporal/pe_frames.py` | unchanged | `e8f0fa059f01109daba7f9440f8d6e5ab5228ed0ae94e68041712b59b6638be5` | `e8f0fa059f01109daba7f9440f8d6e5ab5228ed0ae94e68041712b59b6638be5` |
| `temporal/results/experiment_results.csv` | unchanged | `9eb4e2df9bcc70b0995db20fca1877f722d143c4724cd54375650e54d413ee4b` | `9eb4e2df9bcc70b0995db20fca1877f722d143c4724cd54375650e54d413ee4b` |
| `temporal/results/final_experiment_summary.json` | unchanged | `4b126649d1664eb4645ce0a4da6059dc06aac62eb0c78907c8d322e06472be3a` | `4b126649d1664eb4645ce0a4da6059dc06aac62eb0c78907c8d322e06472be3a` |
| `temporal/results/gru32_seed0_epochs.csv` | unchanged | `343cd07151ee82f5bd36dbd51da0e56677ff45fa18ff2aa1e346bbe527b0070a` | `343cd07151ee82f5bd36dbd51da0e56677ff45fa18ff2aa1e346bbe527b0070a` |
| `temporal/results/gru32_seed0_run.csv` | modified: checkpoint_path only | `5a74ca163fa27dfdb2cf07e6175e52910fbfcf816140244d31e3c67a0d35bfbf` | `53eb3f09cd74a86e81c57f1d41c08566cb26c19457cc648de3bfd3bdc2ea75aa` |
| `temporal/results/gru32_seed1_epochs.csv` | unchanged | `ff25e6353db43bc67caa265cec9e8700d62bf85801c0eb5a23d82d4729dbbd27` | `ff25e6353db43bc67caa265cec9e8700d62bf85801c0eb5a23d82d4729dbbd27` |
| `temporal/results/gru32_seed1_run.csv` | modified: checkpoint_path only | `fbe6722645e2371999398d3aca779257aacca08b280ab26b2cc65c7a44c58956` | `a55ecbbefd2bd142cf7417cd019e9b73d423206420405873bdcb976048cbd47c` |
| `temporal/results/gru32_seed2_epochs.csv` | unchanged | `f63b1e0b0619b1a3cbcafa0bc8489c10ff502b45d4314a36949f541356d89c9c` | `f63b1e0b0619b1a3cbcafa0bc8489c10ff502b45d4314a36949f541356d89c9c` |
| `temporal/results/gru32_seed2_run.csv` | modified: checkpoint_path only | `c7ab0a5aacd01d0ec3ce6beaf83df031f17afc92bfeea1eb064e6069e704b4ac` | `b523bb515081b1d617a6f151a48d7954cb2e4b04cc1ab41ba0d707e21ba73dd0` |
| `temporal/results/gru32_seed3_epochs.csv` | unchanged | `0a2f940156d3caccb2a088002781cc53fa56f1ee7813699d5d5d395257d32bf5` | `0a2f940156d3caccb2a088002781cc53fa56f1ee7813699d5d5d395257d32bf5` |
| `temporal/results/gru32_seed3_run.csv` | modified: checkpoint_path only | `eb448f977fb59c3ff660579c0d4575edbbda7b13e429c6f880569183cbdebce1` | `e15506cca67f3492088ed0c3c6ba693e078280f2e94cfd35382f165587f9a37d` |
| `temporal/results/gru32_seed4_epochs.csv` | unchanged | `dac18f6faec905469f7a30c8b109876f01281c8e904aa4290def6caac4d162b9` | `dac18f6faec905469f7a30c8b109876f01281c8e904aa4290def6caac4d162b9` |
| `temporal/results/gru32_seed4_run.csv` | modified: checkpoint_path only | `845084d9dbb85c84a1ec49be676cbab8d1f0ebfe78dc4a0ed705b4fbb0bfe76c` | `0482f474f246147768aa8858d8a45743678165ac95125f3270508bf6b5a87d46` |
| `temporal/results/gru64_seed0_epochs.csv` | unchanged | `ca9c6dbf6c036e1895570a26bed4808c4929f5eb448bb3bab423ae80ec1d0b3d` | `ca9c6dbf6c036e1895570a26bed4808c4929f5eb448bb3bab423ae80ec1d0b3d` |
| `temporal/results/gru64_seed0_run.csv` | modified: checkpoint_path only | `f6e9558d51d3df9c45cb8f2ca00f1c79716aee355e470304a2db5271622dac42` | `d81f98353558bdb7c1e07901e13fa255eb74474e547676b2833689b100c78296` |
| `temporal/results/gru64_seed1_epochs.csv` | unchanged | `f2683f315897edef83b8b358472ddb17b0e287add1270a1780a1c22da07c0f8a` | `f2683f315897edef83b8b358472ddb17b0e287add1270a1780a1c22da07c0f8a` |
| `temporal/results/gru64_seed1_run.csv` | modified: checkpoint_path only | `ee911d313e6de8064fb9a58bf617315491a35d6dae9dee77b082f4e0a06d25ea` | `d78eff82576f2aec88ab026cb3139d5318d1eccc92dd8055e7245e76ebbb5366` |
| `temporal/results/gru64_seed2_epochs.csv` | unchanged | `7cb3882e966173f3024f4810a2eaa1440878687c0c699e1519f2cf64cee4edfc` | `7cb3882e966173f3024f4810a2eaa1440878687c0c699e1519f2cf64cee4edfc` |
| `temporal/results/gru64_seed2_run.csv` | modified: checkpoint_path only | `48e26a6b06048b5b3a38276186ef6eb0ffd8467ec30d47d6d4912d2b9f7e4285` | `9dc5c2389759668059ca20c64af22014e34eed6849df8bc72448e43dbe672bc5` |
| `temporal/results/gru64_seed3_epochs.csv` | unchanged | `88f17b15ea5eb3cfe746ad10ba9b1155b88b6e3a4ada65c1f190ff7279db7c3c` | `88f17b15ea5eb3cfe746ad10ba9b1155b88b6e3a4ada65c1f190ff7279db7c3c` |
| `temporal/results/gru64_seed3_run.csv` | modified: checkpoint_path only | `eb49f22db212ea7e279104b147d4483dc7a0c4e12fef5625e112b24a18276429` | `94befc65046d0006e00419a388e61a40ecd1bb411d60454193cec834cc12a5ec` |
| `temporal/results/gru64_seed4_epochs.csv` | unchanged | `4f9650f98d93eeb356175b14d14a99a8149702fd08d0ae884c2ac64ce991c5c4` | `4f9650f98d93eeb356175b14d14a99a8149702fd08d0ae884c2ac64ce991c5c4` |
| `temporal/results/gru64_seed4_run.csv` | modified: checkpoint_path only | `b3d32d9b7654a29750defc2db29dd6d1df99f4b691a00c7aa6e0b1ed168a0091` | `3282efed23871a736ed56e434fe7a95f07b80b47fafafe5c2d895c9bc7b6a051` |
| `temporal/results/implementation_verification.json` | unchanged | `fcacb5003095b12cf65206a96e3229f844e0014455a01bc1154165b791431797` | `fcacb5003095b12cf65206a96e3229f844e0014455a01bc1154165b791431797` |
| `temporal/results/pe_preprocessing_all.json` | unchanged | `3851ee08a484def787abccdd5eb60f09ded607b3d017b8ff24e04326361ee95d` | `3851ee08a484def787abccdd5eb60f09ded607b3d017b8ff24e04326361ee95d` |
| `temporal/results/pe_preprocessing_smoke.json` | unchanged | `61e714dc410b58b1d884144bcde299c8d87a5317569f7f90e83c653737de0eab` | `61e714dc410b58b1d884144bcde299c8d87a5317569f7f90e83c653737de0eab` |
| `temporal/results/preprocessing_all.json` | unchanged | `eb7fca1111a915b93a03b7d7b14bc57c5cc22ba8c659462cf55febea58a1bf73` | `eb7fca1111a915b93a03b7d7b14bc57c5cc22ba8c659462cf55febea58a1bf73` |
| `temporal/results/preprocessing_smoke.json` | unchanged | `8cf5db512e81373a2b85114c1d4bd58fb755d699d39f1a25c685684893c314d8` | `8cf5db512e81373a2b85114c1d4bd58fb755d699d39f1a25c685684893c314d8` |
| `temporal/results/selected_configuration.json` | unchanged | `115d5fee2a5aaf464bf6ca562281b9c9af71c1f81361985fffb6ff6737e14671` | `115d5fee2a5aaf464bf6ca562281b9c9af71c1f81361985fffb6ff6737e14671` |
| `temporal/results/test_error_analysis.json` | unchanged | `bbddcffa55337c8c26041c98d2dbd48d5ae9766b6dbb65e570ad919ac4064bdd` | `bbddcffa55337c8c26041c98d2dbd48d5ae9766b6dbb65e570ad919ac4064bdd` |
| `temporal/results/test_integrity.json` | unchanged | `9729bf83e889077ac1e0a29dc905b8cfeb5d583804ca510a5c0d50c0a65577e5` | `9729bf83e889077ac1e0a29dc905b8cfeb5d583804ca510a5c0d50c0a65577e5` |
| `temporal/results/test_metrics.csv` | unchanged | `0fb9a8be33e83760d2c25c64e2c18ed1b145eb453a86245a2cedb35594d7170a` | `0fb9a8be33e83760d2c25c64e2c18ed1b145eb453a86245a2cedb35594d7170a` |
| `temporal/results/test_output_audit.json` | unchanged | `72f51bd679153af2579b10e09d1061bf7ba6251043346a2879c6a20b28956e01` | `72f51bd679153af2579b10e09d1061bf7ba6251043346a2879c6a20b28956e01` |
| `temporal/results/test_pre_evaluation_state.json` | unchanged | `7aa5819845a77436f5f3c411cb2ee7538e0b40e2da5174371e4661f92fa87569` | `7aa5819845a77436f5f3c411cb2ee7538e0b40e2da5174371e4661f92fa87569` |
| `temporal/results/test_predictions.csv` | unchanged | `6228ebd304e2aa7ef5964d51a6995229e032bd2f5cfc5de1123d10317fcf2f44` | `6228ebd304e2aa7ef5964d51a6995229e032bd2f5cfc5de1123d10317fcf2f44` |
| `temporal/results/test_report.md` | unchanged | `f198a909187789dbedcad55aeac9eff77d8d9a36ade6734eeb1690d462ef4117` | `f198a909187789dbedcad55aeac9eff77d8d9a36ade6734eeb1690d462ef4117` |
| `temporal/results/test_summary.csv` | unchanged | `0e668626ce7dfe61c8aee1b71b044a19d0a5f217ea23f575768eca62322989a9` | `0e668626ce7dfe61c8aee1b71b044a19d0a5f217ea23f575768eca62322989a9` |
| `temporal/results/training_validation_audit.json` | unchanged | `060ae96bbd3082f7e5495d15b1f5450f00301643b2814a66367c913f8b58d14b` | `060ae96bbd3082f7e5495d15b1f5450f00301643b2814a66367c913f8b58d14b` |
| `temporal/results/validation_report.md` | unchanged | `f760a7dcb7b8050092404c12290127200d7f1f9fc7e133a7ba63c6945a85d886` | `f760a7dcb7b8050092404c12290127200d7f1f9fc7e133a7ba63c6945a85d886` |
| `temporal/results/validation_summary.csv` | unchanged | `ad203cb574128ae7236811d1a72b2af743fff41297149b05d13aa2b8187fc631` | `ad203cb574128ae7236811d1a72b2af743fff41297149b05d13aa2b8187fc631` |
| `temporal/results_pe/FINAL_REPORT.md` | unchanged | `63c01b4aec53918da4cd438b4559007867fb3793f367402d8d4f3b9e42ad5184` | `63c01b4aec53918da4cd438b4559007867fb3793f367402d8d4f3b9e42ad5184` |
| `temporal/results_pe/celebdf_gru_predictions.csv` | unchanged | `7b35db51575e921a6c253e068a6946f6885ca0959d8f9b78101a5e912250a631` | `7b35db51575e921a6c253e068a6946f6885ca0959d8f9b78101a5e912250a631` |
| `temporal/results_pe/celebdf_gru_summary.csv` | unchanged | `adf4d7012a43ca0e749f9b206f087daf398e0cf2a1a55bbdcf08fb95ab8fe8a2` | `adf4d7012a43ca0e749f9b206f087daf398e0cf2a1a55bbdcf08fb95ab8fe8a2` |
| `temporal/results_pe/final_comparison_celebdf.csv` | unchanged | `236f79ffc0a6c9d132cde880ae6c2db0758b18f9eef8fd9d99b202a3a2afbcf7` | `236f79ffc0a6c9d132cde880ae6c2db0758b18f9eef8fd9d99b202a3a2afbcf7` |
| `temporal/results_pe/final_comparison_ffpp.csv` | unchanged | `2cfdac1269bf48ae9b10e779b1b0e010e8c21cd20c99bbb6bab97a48738c1489` | `2cfdac1269bf48ae9b10e779b1b0e010e8c21cd20c99bbb6bab97a48738c1489` |
| `temporal/results_pe/final_comparison_gains.csv` | unchanged | `d57b96ab344e2e412eee31efca10fa7ab23f559f3fe36c988e6900762fa08b81` | `d57b96ab344e2e412eee31efca10fa7ab23f559f3fe36c988e6900762fa08b81` |
| `temporal/results_pe/gru32_pe_seed0_epochs.csv` | unchanged | `fb4b8afc59035c6f3a433890f301308d0b083af3eba610fe8a2a823b50795bf9` | `fb4b8afc59035c6f3a433890f301308d0b083af3eba610fe8a2a823b50795bf9` |
| `temporal/results_pe/gru32_pe_seed1_epochs.csv` | unchanged | `d8e5ee0312c6a811e4aab6dd8ec3c7b2e42e80aded630c1b2d36aaa3dddb060b` | `d8e5ee0312c6a811e4aab6dd8ec3c7b2e42e80aded630c1b2d36aaa3dddb060b` |
| `temporal/results_pe/gru32_pe_seed2_epochs.csv` | unchanged | `01a38b2d215a31ca89be176c7193514fe132f0e68be57bbcc522150abe1259b2` | `01a38b2d215a31ca89be176c7193514fe132f0e68be57bbcc522150abe1259b2` |
| `temporal/results_pe/gru32_pe_seed3_epochs.csv` | unchanged | `ba4b1eeeb2fadf1f1eb323b03011ea3541131377836cd32c963742d68d66455c` | `ba4b1eeeb2fadf1f1eb323b03011ea3541131377836cd32c963742d68d66455c` |
| `temporal/results_pe/gru32_pe_seed4_epochs.csv` | unchanged | `86204bfa546e7a453658a737bc2fb2675c80e47682688c5b6f7598f76454e43d` | `86204bfa546e7a453658a737bc2fb2675c80e47682688c5b6f7598f76454e43d` |
| `temporal/results_pe/gru32_pe_test_predictions.csv` | unchanged | `1e9793ea015b56e3a6a972b2b8b8a90fa8e7cd32af7412ac5c22ab61eb8add96` | `1e9793ea015b56e3a6a972b2b8b8a90fa8e7cd32af7412ac5c22ab61eb8add96` |
| `temporal/results_pe/gru32_pe_test_summary.csv` | unchanged | `709090314df6843c61ffbfeb2096b25c8116f8dca82dd001d0e148e858b7074a` | `709090314df6843c61ffbfeb2096b25c8116f8dca82dd001d0e148e858b7074a` |
| `temporal/results_pe/gru32_pe_validation_predictions.csv` | unchanged | `adf430da017c2f2bc6e12aa0e2d593fa298f140ffc8ddb1e2d7bc83151eb4c14` | `adf430da017c2f2bc6e12aa0e2d593fa298f140ffc8ddb1e2d7bc83151eb4c14` |
| `temporal/results_pe/gru32_pe_validation_summary.csv` | unchanged | `7be3601aaafa8bc957c2a2ea9be2ce19ee3f934d3ad1e3d7aa694b5504df7c9a` | `7be3601aaafa8bc957c2a2ea9be2ce19ee3f934d3ad1e3d7aa694b5504df7c9a` |
| `temporal/test_dataset_split.py` | unchanged | `f248724ff7539ddb4e97c2db032c421f2fc8397b41a617844a1fa719b90d9554` | `f248724ff7539ddb4e97c2db032c421f2fc8397b41a617844a1fa719b90d9554` |
| `temporal/test_feature_dataset.py` | unchanged | `4f51de4e0db9ba74b750987bad28eb9b8f3856b7fddc9f0e36ef8078841949a0` | `4f51de4e0db9ba74b750987bad28eb9b8f3856b7fddc9f0e36ef8078841949a0` |
| `temporal/test_gend_features.py` | unchanged | `a22860cb20a79d7996b775cf7e78c8d9767e435c20a704aba3ec95af070403d9` | `a22860cb20a79d7996b775cf7e78c8d9767e435c20a704aba3ec95af070403d9` |
| `temporal/test_gru.py` | unchanged | `e8dcf7fc66da7646782d471c8741198150386ff4474032c22eda4cc4d1a21422` | `e8dcf7fc66da7646782d471c8741198150386ff4474032c22eda4cc4d1a21422` |
| `temporal/test_gru_forward.py` | unchanged | `784f680c138294fddb799c5cbd6f7ec58ce540546b8f81edde10f4b01eb0c90a` | `784f680c138294fddb799c5cbd6f7ec58ce540546b8f81edde10f4b01eb0c90a` |
| `temporal/test_reporting.py` | unchanged | `1a175fdf42fd9188697c4d0fe7116bdcd50015686a798a3fa35ace4288425d27` | `1a175fdf42fd9188697c4d0fe7116bdcd50015686a798a3fa35ace4288425d27` |
| `temporal/test_video_loader.py` | unchanged | `1a71b930b22660cbc6554afbe38b3692c81d9a6e4e31b8dd7413b36f7135e274` | `1a71b930b22660cbc6554afbe38b3692c81d9a6e4e31b8dd7413b36f7135e274` |
| `temporal/train_gru.py` | unchanged | `b6bc951441281da22006e6c7720b52ca8cdbef761484fb6e7ac1f00a5089214c` | `b6bc951441281da22006e6c7720b52ca8cdbef761484fb6e7ac1f00a5089214c` |
| `temporal/train_large_gru.py` | unchanged | `248fe1fb0d488b5d531214ceae9d3859819bb0241cfd352b4b7d47ca0fd294f3` | `248fe1fb0d488b5d531214ceae9d3859819bb0241cfd352b4b7d47ca0fd294f3` |
| `temporal/train_pe_gru.py` | unchanged | `879ba01580c1835c4ed36cac1d056f9f181731680327cf36d069b5fb06c70d7a` | `879ba01580c1835c4ed36cac1d056f9f181731680327cf36d069b5fb06c70d7a` |
| `temporal/verify_large_experiment.py` | unchanged | `8043adcdfcdd7c64399a0f9d2d2ba7261450dfc82a82b403ded992db106e4338` | `8043adcdfcdd7c64399a0f9d2d2ba7261450dfc82a82b403ded992db106e4338` |
| `temporal/video_loader.py` | unchanged | `ce339cb82e0d9fd2db17fdfba0d1dbc6dc2923878efb54f3c12bcdeb869a951c` | `ce339cb82e0d9fd2db17fdfba0d1dbc6dc2923878efb54f3c12bcdeb869a951c` |
| `docs/PUBLICATION_COPY_AUDIT.md` | added: this report | absent | self-reference excluded |

## Verification and regression checks

- Final file inventory: **188**. Exactly one removed file (media/gradio.png), ten modified CSVs and this one added audit. No other original file differs.
- All 94 Python source files passed in-memory syntax compilation; no bytecode files were written into the copy.
- Both screenshots and all five graphs decoded successfully with Pillow image verification (seven images total).
- All seven local README image references resolve to the expected retained files. Four remote badge references were identified but not downloaded.
- The media/ directory and .git directory are absent. No video, tensor, model-checkpoint or ONNX files are present.
- All 54 result files remain; every non-checkpoint_path CSV field was compared as an exact string, preserving numerical representations as well as values, row order and columns.
- Both manifests, the DVC receipt/configuration and all protected source/frontend/license/graph/screenshot bytes match the reference ZIP. No model weights or data were substituted.
- python -B -m temporal.verify_large_experiment: PASS (split, corruption rejection, shared indices, synthetic backward-pass contract, encoder pass, metrics and checkpoint names). This is not retraining or full inference.
- python -B -m unittest discover -s frontend/tests -v: 14 tests, 10 passed, 3 pre-existing Gradio compatibility errors (Video rejects buttons keyword), 1 skip because FFmpeg is unavailable. No frontend code was changed to hide these baseline environment issues.
- Credential-pattern screening found no private-key blocks, common GitHub tokens, AWS access keys, embedded credential URLs or obvious literal credential assignments. This is not exhaustive clearance.
- After audit redaction, no personal absolute filesystem paths remain in publication files. Two unchanged commented upstream /media/... examples in src/model/forada/config.yaml remain as non-runtime source examples; they are not the personal research paths. Historical media mentions are explanatory, not broken runtime/image references.
- Reference ZIP SHA-256 remains 559df0bf333127c050ce9791860c526c968d49d083f7ef0b160c714954900e14. The original research checkout remains clean at commit 896055860fac3cf2b0ebca929ebbaa060d9cf7ef. No writes were made to original research assets or the backup.
- No Git initialization, new commit, GitHub operation, history rewrite, DVC operation or upload was performed. Screenshot redistribution rights remain a separate release-review prerequisite; the local audit redaction is complete.
