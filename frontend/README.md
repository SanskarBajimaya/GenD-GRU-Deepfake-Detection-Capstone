# GenD Deepfake Intelligence frontend

Private, local-first Gradio 6 dashboard for the released GenD visual heads and the capstone five-seed GRU-32 models.

## Architecture

- `app.py`: responsive dashboard and UI callbacks.
- `preprocessing.py`: MP4 validation and the original RetinaFace/alignment workflow.
- `model_loader.py`: lazy visual encoders and strict GRU checkpoint validation.
- `predictor.py`: genuine inference, seed averaging, and score-level ensembles.
- `repository.py`: future persistence boundary. The default saves nothing.
- `style.css`: premium dark dashboard theme.

No upload is copied into the repository and no result is persisted by default.

## Install and launch

Use one Python environment for both Gradio and inference. Python 3.12 is the repository's documented environment and is the safest choice for the pinned research stack.

```powershell
python -m pip install -r frontend/requirements.txt
python frontend/app.py
```

Open <http://127.0.0.1:7860>. The app binds to localhost and never enables Gradio sharing.

Optional limits:

```powershell
$env:GEN_D_MAX_UPLOAD_MB = "500"
$env:GEN_D_MAX_DURATION_SECONDS = "300"
$env:GEN_D_PORT = "7860"
python frontend/app.py
```

If Python 3.13 cannot resolve a pinned PyTorch or ONNX Runtime wheel, create a Python 3.12 virtual environment rather than changing research dependency versions without validation.

The root research environment pins Transformers 4.56.2, which requires `huggingface-hub < 1.0`. Gradio 6.31 requires `huggingface-hub >= 1.16`, so those exact packages cannot coexist. The isolated frontend requirements use compatible Transformers 5 while leaving all research source, preprocessing, model identifiers, and weights unchanged. The frontend loader supplies a small compatibility subclass because Transformers 5 defaults outer models to a meta device, while legacy GenD loads its backbone inside `__init__`. It also restores CLIP's standard non-persistent sequential position-ID buffer if the v5 custom-checkpoint loader leaves that buffer uninitialized; learned weights are not changed. Keep training/reproduction work in its original environment.

## Models and assets

The first visual-model run downloads the official Hugging Face model unless cached:

- `yermandy/GenD_CLIP_L_14`
- `yermandy/GenD_PE_L`

RetinaFace uses `weights/models/buffalo_l/det_10g.onnx`, matching the existing pipeline. If it is absent, the frontend downloads the detector from the repository's commit-pinned Hugging Face URL with a 300 MB safety ceiling and an atomic final rename. It never invokes `wget` or interpolates an uploaded filename into a command.

| Configuration | Required assets | Uploaded-video score |
|---|---|---|
| GenD-CLIP | Released HF model | Mean of 32 frame-level FAKE probabilities |
| GenD-PE | Released HF model | Mean of 32 frame-level FAKE probabilities |
| CLIP-GRU-32 | CLIP + five CLIP GRUs | Mean across seeds 0–4 |
| PE-GRU-32 | PE + five PE GRUs | Mean across seeds 0–4 |
| GenD CLIP + PE ensemble | Both released models | Mean of the two video probabilities |
| CLIP-GRU + PE-GRU ensemble | Both backbones + ten GRUs | Mean of the two five-seed probabilities |

Required GRU paths:

```text
temporal/models/gru32_seed0_best.pt ... gru32_seed4_best.pt
temporal/models_pe/gru32_pe_seed0_best.pt ... gru32_pe_seed4_best.pt
```

GRU choices are disabled unless every checkpoint for that configuration exists. Each file is safely loaded with `weights_only=True` and must match the repository manifest and split hashes, feature size 1,024, hidden size 128, one layer, 32 frames, split seed 2026, expected training seed, and `completed=True`.

The tracked `temporal/models/best_gru.pt` is the old 16-frame pilot and is intentionally never substituted.

## Exact inference behavior

The service reuses `temporal.video_loader.load_aligned_frame_sequence`:

1. The original max-spread sampler searches for 64 valid faces.
2. RetinaFace chooses the largest face.
3. Five landmarks align it to 224 × 224 with scale 1.3.
4. Frames are restored to chronological order.
5. Every second face is selected, matching GRU-32 input `[1, 32, 1024]`.
6. CLIP and PE process the same aligned images.

Released heads use their real pretrained classification weights. Softmax class index 1 is `FAKE`, and the original video score is the mean of frame probabilities. GRUs use sigmoid logits and their five seeds are averaged exactly as in `temporal/eval_celebdf_gru.py`.

## GPU and CPU

CUDA is selected automatically. Only one large visual encoder remains resident to reduce peak memory. If CUDA runs out of memory, the analysis retries once on CPU. CPU is valid but slow, especially for Compare Available.

The queue runs one inference job at a time with at most four waiting jobs.

## Security and privacy

- MP4 extension, size, duration, stream metadata, and first-frame decoding are validated.
- Default limits are 500 MB and five minutes.
- The server is localhost-only with `share=False`.
- Research data, weights, runs, and model directories are blocked from file serving.
- Callback APIs are private and monitoring is disabled.
- Full temporary paths are not returned to the browser.
- Uploaded filenames never enter shell commands.
- No upload or prediction is persisted by default.

Gradio owns the upload temporary file; cache cleanup is configured hourly. Before any multi-user or internet deployment, add authentication, a reverse proxy, malware scanning, tenant isolation, and an approved retention policy.

## Benchmark panel

The Celeb-DF v2 AUROC panel is verified against `temporal/results_pe/final_comparison_celebdf.csv`. It is clearly historical dataset-level research context and is never combined with an uploaded-video prediction.

## Tests

```powershell
python -m unittest discover -s frontend/tests -v
```

Lightweight tests cover request planning, genuine-result rendering, checkpoint naming/availability, bounded RetinaFace provisioning, upload rejection, and Gradio construction. Real inference additionally requires PyTorch/video dependencies, RetinaFace weights, Hugging Face cache or network access, and exact GRU checkpoints for temporal modes.

Validated in this workspace on 2026-11-03 with Python 3.13 and CPU-only PyTorch:

- All 11 unit tests passed; Gradio emitted non-fatal unclosed-event-loop `ResourceWarning` messages during teardown.
- The app served successfully on `127.0.0.1:7860` and returned HTTP 200.
- A temporary synthetic MP4 passed metadata/decoder validation, and its intentional no-face condition returned a clear error with no prediction.
- The pinned RetinaFace ONNX model loaded with `CPUExecutionProvider`.
- Official GenD-CLIP and GenD-PE weights loaded and completed one-frame forward smoke tests with feature shape `[1, 1024]`, logit shape `[1, 2]`, and probabilities summing to 1.
- End-to-end prediction on a real face video was not run because no test video is included in the repository.
- GRU-32 execution was not run because the repository does not contain the required five CLIP and five PE checkpoints.

## Troubleshooting

- **Missing PyTorch or codec:** install `frontend/requirements.txt` into the launch interpreter.
- **Hugging Face model cannot load:** provide network access or pre-populate its cache.
- **GRU choice absent:** provide all five exact checkpoints for that backbone, then restart.
- **Not enough faces:** use a video with one clear face for at least 64 frames.
- **CPU is slow:** configure compatible CUDA; do not reduce frame count because that changes the trained pipeline.
- **Port occupied:** set `GEN_D_PORT` to another local port.

## Future SQL integration

Implement `AnalysisRepository.save()` in `repository.py` and inject it into `PredictionService`. Store metadata and model outputs—not raw videos—unless an explicit retention policy is approved. A later database branch can add user/session IDs, job status, timestamps, model/checkpoint versions, and external-storage object keys without coupling SQL to the UI.
