"""Orchestrate released GenD heads and validated five-seed GRU inference."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from model_loader import MODEL_LABELS, InvalidCheckpointError, MissingCheckpointError, ModelManager, RuntimeDependencyError
from preprocessing import FaceExtractionError, VideoMetadata, extract_research_frames
from repository import AnalysisRepository, NullAnalysisRepository


class AnalysisError(RuntimeError): pass


@dataclass(frozen=True)
class Prediction:
    key: str
    label: str
    fake_probability: float


@dataclass(frozen=True)
class AnalysisReport:
    selected: Prediction
    predictions: tuple[Prediction, ...]
    processing_seconds: float
    frames_used: int
    device: str
    notes: tuple[str, ...]


class PredictionService:
    def __init__(self, manager: ModelManager, repository: AnalysisRepository | None = None):
        self.manager = manager
        self.repository = repository or NullAnalysisRepository()

    @staticmethod
    def _plan(selected: str, compare_all: bool, availability: dict[str, tuple[bool, str]]) -> list[str]:
        if selected not in MODEL_LABELS:
            raise AnalysisError("Choose a valid detection configuration.")
        if not availability[selected][0]:
            raise AnalysisError(f"{MODEL_LABELS[selected]} is unavailable: {availability[selected][1]}.")
        return [key for key in MODEL_LABELS if availability[key][0]] if compare_all else [selected]

    @staticmethod
    def _required_backbones(plan: list[str]) -> list[str]:
        clip = any(k in plan for k in ("gend_clip","clip_gru","head_ensemble","gru_ensemble"))
        pe = any(k in plan for k in ("gend_pe","pe_gru","head_ensemble","gru_ensemble"))
        return [key for key, needed in (("clip",clip),("pe",pe)) if needed]

    def _encode(self, backbone: str, frames: list, progress: Callable[[float,str],None], base: float):
        torch = self.manager._load_torch()
        model = self.manager.get_gend(backbone)
        features, logits = [], []
        batch_size = 8 if self.manager.device.type == "cuda" else 4
        with torch.inference_mode():
            for start in range(0,len(frames),batch_size):
                current = frames[start:start+batch_size]
                batch = torch.stack([model.feature_extractor.preprocess(frame) for frame in current]).to(self.manager.device)
                encoded = model.feature_extractor(batch)
                features.append(encoded.float().cpu())
                logits.append(model.model.forward(encoded).float().cpu())
                progress(base + .20*min((start+len(current))/len(frames),1), f"Encoding {backbone.upper()} faces")
        return torch.cat(features), torch.cat(logits)

    def _gru_probability(self, backbone: str, features, progress: Callable[[float,str],None], stage: float) -> float:
        torch = self.manager._load_torch()
        values = []
        with torch.inference_mode():
            sequence = features.unsqueeze(0).to(self.manager.device)
            for index, model in enumerate(self.manager.get_grus(backbone),1):
                values.append(float(model(sequence).sigmoid().item()))
                progress(stage+index*.01, f"Scoring {backbone.upper()} GRU seed {index}/5")
        return sum(values)/len(values)

    def _run_models(self, frames: list, plan: list[str], progress: Callable[[float,str],None]) -> dict[str,float]:
        torch = self.manager._load_torch()
        values: dict[str,float] = {}
        for index, backbone in enumerate(self._required_backbones(plan)):
            base = .42 + index*.24
            features, logits = self._encode(backbone,frames,progress,base)
            values[f"gend_{backbone}"] = float(logits.softmax(dim=-1)[:,1].mean().item())
            if f"{backbone}_gru" in plan or "gru_ensemble" in plan:
                values[f"{backbone}_gru"] = self._gru_probability(backbone,features,progress,base+.20)
            del features, logits
            self.manager.release_encoder()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        if "head_ensemble" in plan: values["head_ensemble"] = (values["gend_clip"]+values["gend_pe"])/2
        if "gru_ensemble" in plan: values["gru_ensemble"] = (values["clip_gru"]+values["pe_gru"])/2
        return values

    @staticmethod
    def _is_cuda_oom(exc: Exception) -> bool:
        text = str(exc).lower()
        return "cuda" in text and "out of memory" in text

    def analyze(self, video_path: str|Path, selected_model: str, compare_all: bool, metadata: VideoMetadata, progress: Callable[[float,str],None]) -> AnalysisReport:
        plan = self._plan(selected_model,compare_all,self.manager.availability())
        started = time.perf_counter()
        try:
            progress(.12,"Loading RetinaFace")
            detector = self.manager.get_face_detector()
            progress(.18,"Sampling 64 valid face frames")
            frames, ids = extract_research_frames(Path(video_path),detector)
            progress(.40,"Face sequence aligned chronologically")
            try: values = self._run_models(frames,plan,progress)
            except Exception as exc:
                if not self._is_cuda_oom(exc): raise
                progress(.42,"GPU memory exhausted; retrying safely on CPU")
                self.manager.fallback_to_cpu()
                values = self._run_models(frames,plan,progress)
        except FaceExtractionError as exc: raise AnalysisError(str(exc)) from exc
        except (RuntimeDependencyError,MissingCheckpointError,InvalidCheckpointError) as exc: raise AnalysisError(str(exc)) from exc
        except Exception as exc:
            if self._is_cuda_oom(exc): raise AnalysisError("Inference exceeded available GPU and CPU memory.") from exc
            raise
        ordered = tuple(Prediction(key,MODEL_LABELS[key],values[key]) for key in plan if key in values)
        selected = next((item for item in ordered if item.key == selected_model),ordered[0])
        notes = [
            "Positive class is FAKE; 50% is the display decision threshold, not a calibrated guarantee.",
            "Faces were aligned at 224 × 224 with scale 1.3 using the original RetinaFace pipeline.",
            f"GRU-compatible sampling used 32 chronological faces from frame IDs {ids[0]}–{ids[-1]}.",
            "Uploads and predictions are not persisted by the frontend.",
        ]
        if len(plan)<len(MODEL_LABELS) and compare_all:
            notes.append("Unavailable GRU configurations were omitted because their five-seed checkpoints are incomplete.")
        report = AnalysisReport(selected,ordered,time.perf_counter()-started,32,str(self.manager.device).upper(),tuple(notes))
        self.repository.save({"filename":metadata.safe_name,"model":selected.key,"fake_probability":selected.fake_probability,"frames_used":32,"processing_seconds":report.processing_seconds})
        progress(.99,"Preparing analytical report")
        return report


__all__ = ["MODEL_LABELS","AnalysisError","AnalysisReport","Prediction","PredictionService"]
