"""Lazy, validated model loading without heavy ML imports at UI startup."""

from __future__ import annotations

import gc
import os
from contextlib import nullcontext
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.request import Request, urlopen

HF_MODEL_IDS = {"clip": "yermandy/GenD_CLIP_L_14", "pe": "yermandy/GenD_PE_L"}
RETINAFACE_URL = (
    "https://huggingface.co/datasets/theanhntp/Liblib/resolve/"
    "ae4357741af379482690fe3e0f2fa6fd32ba33b4/"
    "insightface/models/buffalo_l/det_10g.onnx"
)
MAX_RETINAFACE_BYTES = 300 * 1024 * 1024
MODEL_LABELS = {
    "gend_clip": "GenD-CLIP", "gend_pe": "GenD-PE", "clip_gru": "CLIP-GRU-32",
    "pe_gru": "PE-GRU-32", "head_ensemble": "GenD CLIP + PE ensemble",
    "gru_ensemble": "CLIP-GRU + PE-GRU ensemble",
}


class RuntimeDependencyError(RuntimeError): pass
class MissingCheckpointError(RuntimeError): pass
class InvalidCheckpointError(RuntimeError): pass


class ModelManager:
    """Own heavyweight objects and keep at most one visual encoder resident."""

    def __init__(self, project_root: Path):
        self.root = Path(project_root).resolve()
        self.lock = RLock()
        self._torch = self._device = self._face_detector = None
        self._encoder_key: str | None = None
        self._encoder = None
        self._gru_cache: dict[str, list[Any]] = {}

    def _load_torch(self):
        if self._torch is not None:
            return self._torch
        try:
            import torch
        except ImportError as exc:
            raise RuntimeDependencyError("PyTorch is not installed in this Python environment. Install frontend/requirements.txt.") from exc
        self._torch = torch
        return torch

    @property
    def device(self):
        torch = self._load_torch()
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    @property
    def device_label(self) -> str:
        try:
            return "GPU" if self._load_torch().cuda.is_available() else "CPU"
        except RuntimeDependencyError:
            return "RUNTIME CHECK"

    def checkpoint_paths(self, backbone: str) -> list[Path]:
        if backbone == "clip":
            return [self.root/"temporal"/"models"/f"gru32_seed{i}_best.pt" for i in range(5)]
        if backbone == "pe":
            return [self.root/"temporal"/"models_pe"/f"gru32_pe_seed{i}_best.pt" for i in range(5)]
        raise ValueError(f"Unknown GRU backbone: {backbone}")

    def missing_checkpoints(self, backbone: str) -> list[Path]:
        return [path for path in self.checkpoint_paths(backbone) if not path.is_file()]

    def availability(self) -> dict[str, tuple[bool, str]]:
        clip, pe = self.missing_checkpoints("clip"), self.missing_checkpoints("pe")
        return {
            "gend_clip": (True, "Released Hugging Face head · downloads on first use"),
            "gend_pe": (True, "Released Hugging Face head · downloads on first use"),
            "clip_gru": (not clip, "Five checkpoints verified" if not clip else f"Missing {len(clip)}/5 checkpoints"),
            "pe_gru": (not pe, "Five checkpoints verified" if not pe else f"Missing {len(pe)}/5 checkpoints"),
            "head_ensemble": (True, "Runs both released visual heads"),
            "gru_ensemble": (not clip and not pe, "Ten checkpoints verified" if not clip and not pe else f"Missing {len(clip)+len(pe)}/10 checkpoints"),
        }

    def _ensure_face_detector_weight(self) -> Path:
        """Provision the repository's pinned detector without invoking a shell."""
        target = self.root / "weights" / "models" / "buffalo_l" / "det_10g.onnx"
        if target.is_file() and target.stat().st_size > 1024 * 1024:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".onnx.partial")
        try:
            request = Request(RETINAFACE_URL, headers={"User-Agent": "GenD-Capstone/1.0"})
            total = 0
            with urlopen(request, timeout=60) as response, partial.open("wb") as output:
                declared = int(response.headers.get("Content-Length") or 0)
                if declared > MAX_RETINAFACE_BYTES:
                    raise RuntimeDependencyError("The RetinaFace download exceeded the configured safety limit.")
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_RETINAFACE_BYTES:
                        raise RuntimeDependencyError("The RetinaFace download exceeded the configured safety limit.")
                    output.write(chunk)
            if total < 1024 * 1024:
                raise RuntimeDependencyError("The RetinaFace download was incomplete.")
            os.replace(partial, target)
        except RuntimeDependencyError:
            partial.unlink(missing_ok=True)
            raise
        except Exception as exc:
            partial.unlink(missing_ok=True)
            raise RuntimeDependencyError(
                "Could not download the pinned RetinaFace detector. Check network access and retry."
            ) from exc
        return target

    def get_face_detector(self):
        with self.lock:
            if self._face_detector is None:
                model_path = self._ensure_face_detector_weight()
                try:
                    import onnxruntime
                    from src.retinaface import RetinaFace, cuda_available
                except ImportError as exc:
                    raise RuntimeDependencyError("Face-detection dependencies are missing. Install frontend/requirements.txt.") from exc
                if cuda_available() and hasattr(onnxruntime, "preload_dlls"):
                    try:
                        onnxruntime.preload_dlls()
                    except (ImportError, OSError, RuntimeError):
                        pass
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda_available() else ["CPUExecutionProvider"]
                detector = RetinaFace(str(model_path), providers=providers)
                detector.prepare(ctx_id=0, nms_thresh=.4, input_size=(640, 640), det_thresh=.1)
                self._face_detector = detector
            return self._face_detector

    def get_gend(self, backbone: str):
        if backbone not in HF_MODEL_IDS:
            raise ValueError(f"Unknown GenD backbone: {backbone}")
        with self.lock:
            if self._encoder_key == backbone and self._encoder is not None:
                return self._encoder
            self.release_encoder()
            try:
                from src.hf.modeling_gend import GenD
            except ImportError as exc:
                raise RuntimeDependencyError("GenD dependencies are missing. Install frontend/requirements.txt.") from exc

            class FrontendGenD(GenD):
                """Adapt the legacy nested model constructor to Transformers 5."""

                @classmethod
                def get_init_context(cls, *args, **kwargs):
                    # Transformers 5 otherwise constructs the outer model on the
                    # meta device, where GenD's nested backbone load is invalid.
                    return [nullcontext()]

                def __init__(self, config):
                    super().__init__(config)
                    # GenD predates the v5 tied-weight bookkeeping initialized by
                    # post_init(); its CLIP vision submodel does not share weights.
                    self.all_tied_weights_keys = {}

            try:
                model, info = FrontendGenD.from_pretrained(HF_MODEL_IDS[backbone], output_loading_info=True)
            except Exception as exc:
                raise RuntimeDependencyError(f"Could not load {HF_MODEL_IDS[backbone]}. Check network/model cache and Hugging Face access.") from exc
            if any(info.get(k) for k in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs")):
                raise RuntimeDependencyError(f"The released {backbone.upper()} model did not load completely.")
            if backbone == "clip":
                # Transformers 5 can leave this non-persistent legacy buffer
                # uninitialized after loading the outer custom checkpoint.
                # Restore CLIP's canonical arange buffer; no learned weight is
                # changed and the values match CLIPEmbeddings.__init__.
                torch = self._load_torch()
                embeddings = model.feature_extractor.vision_model.embeddings
                positions = embeddings.position_embedding.num_embeddings
                expected_ids = torch.arange(positions, device=embeddings.position_embedding.weight.device).expand(1, -1)
                if embeddings.position_ids.shape != expected_ids.shape or not torch.equal(embeddings.position_ids, expected_ids):
                    embeddings.position_ids = expected_ids
            self._encoder_key, self._encoder = backbone, model.to(self.device).eval().requires_grad_(False)
            return self._encoder

    @staticmethod
    def _validate_checkpoint(checkpoint: dict[str, Any], backbone: str, seed: int, manifest_hash: str, split_hash: str) -> None:
        expected = {"feature_dim":1024,"hidden_dim":128,"num_layers":1,"dropout":0.0,"sequence_length":32,
                    "training_seed":seed,"split_seed":2026,"completed":True,"manifest_hash":manifest_hash,"split_hash":split_hash}
        bad = [key for key, value in expected.items() if checkpoint.get(key) != value]
        if "model_state_dict" not in checkpoint: bad.append("model_state_dict")
        if bad:
            raise InvalidCheckpointError(f"{backbone.upper()} GRU seed {seed} checkpoint is incompatible: {', '.join(bad)}.")

    def get_grus(self, backbone: str) -> list[Any]:
        with self.lock:
            if backbone in self._gru_cache:
                return self._gru_cache[backbone]
            missing = self.missing_checkpoints(backbone)
            if missing:
                raise MissingCheckpointError(f"{MODEL_LABELS[backbone+'_gru']} is unavailable. Missing: {', '.join(p.name for p in missing)}")
            torch = self._load_torch()
            from temporal.build_manifest import fingerprint, load_definitions
            from temporal.model_a import TemporalGRU
            manifest, split = load_definitions()
            mh, sh = fingerprint(manifest), fingerprint(split)
            models = []
            for seed, path in enumerate(self.checkpoint_paths(backbone)):
                try: checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                except Exception as exc: raise InvalidCheckpointError(f"Could not safely load {path.name}.") from exc
                self._validate_checkpoint(checkpoint, backbone, seed, mh, sh)
                model = TemporalGRU(1024,128,num_layers=1,dropout=0.0).to(self.device).eval()
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                models.append(model)
            self._gru_cache[backbone] = models
            return models

    def release_encoder(self) -> None:
        self._encoder = None; self._encoder_key = None; gc.collect()
        if self._torch is not None and self._torch.cuda.is_available(): self._torch.cuda.empty_cache()

    def fallback_to_cpu(self) -> None:
        torch = self._load_torch(); self.release_encoder(); self._gru_cache.clear()
        self._device = torch.device("cpu"); gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
