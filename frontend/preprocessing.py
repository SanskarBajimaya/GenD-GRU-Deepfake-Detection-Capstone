"""Secure upload validation and research-matched facial frame extraction."""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)


class UploadValidationError(ValueError):
    """Raised when an uploaded file is unsafe or cannot be decoded as a video."""


class FaceExtractionError(RuntimeError):
    """Raised when the research pipeline cannot produce enough aligned faces."""


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    safe_name: str
    size_mb: float
    duration_seconds: float
    width: int
    height: int
    frame_count: int
    fps: float


def _path_from_gradio(value: Any) -> Path:
    if isinstance(value, dict):
        value = value.get("path") or value.get("name")
    if hasattr(value, "path"):
        value = value.path
    if hasattr(value, "name") and not isinstance(value, (str, Path)):
        value = value.name
    if not value:
        raise UploadValidationError("No video was supplied.")
    return Path(str(value)).resolve()


def _ffprobe_diagnostics(path: Path) -> None:
    """Log bounded, non-authoritative FFprobe details when it is installed."""
    executable = shutil.which("ffprobe")
    if executable is None:
        LOGGER.debug("FFprobe is not installed; skipping supplemental probe for %s", path.name)
        return
    try:
        result = subprocess.run(
            [
                executable, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames:format=format_name,duration",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if result.returncode:
            LOGGER.warning("FFprobe could not read upload %s: %s", path.name, result.stderr.strip()[:500])
            return
        payload = json.loads(result.stdout)
        stream = (payload.get("streams") or [{}])[0]
        container = payload.get("format") or {}
        LOGGER.info(
            "FFprobe upload=%s container=%s codec=%s dimensions=%sx%s duration=%s frames=%s",
            path.name, container.get("format_name"), stream.get("codec_name"),
            stream.get("width"), stream.get("height"), container.get("duration"), stream.get("nb_frames"),
        )
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        LOGGER.warning("FFprobe diagnostics failed for %s: %s", path.name, exc)


def inspect_video(value: Any, max_size_mb: int, max_duration_seconds: float) -> VideoMetadata:
    """Validate the real MP4 container and the decoder used by the research pipeline."""
    path = _path_from_gradio(value)
    LOGGER.info("Validating Gradio video path=%s", path)
    if path.suffix.lower() != ".mp4":
        LOGGER.warning("Upload rejected: unsupported extension path=%s", path)
        raise UploadValidationError("Only MP4 video uploads are accepted.")
    if not path.is_file():
        LOGGER.warning("Upload rejected: temporary file is missing path=%s", path)
        raise UploadValidationError("The uploaded video is no longer available.")
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb <= 0:
        LOGGER.warning("Upload rejected: empty file path=%s", path)
        raise UploadValidationError("The uploaded file is empty.")
    if size_mb > max_size_mb:
        LOGGER.warning("Upload rejected: file is %.1f MB (limit %s MB) path=%s", size_mb, max_size_mb, path)
        raise UploadValidationError(f"Video is {size_mb:.1f} MB; the limit is {max_size_mb} MB.")

    try:
        import av
        import imageio.v3 as iio
    except ImportError as exc:
        LOGGER.exception("Upload validation dependencies are missing")
        raise UploadValidationError("Video runtime dependencies are missing. Install frontend/requirements.txt.") from exc

    # Opening/demuxing and decoding are separate stages so a corrupt container is
    # not reported as a codec or frame problem.
    try:
        container = av.open(str(path), mode="r")
    except Exception as exc:
        LOGGER.warning("Upload rejected: invalid/unreadable container path=%s error=%r", path, exc)
        _ffprobe_diagnostics(path)
        raise UploadValidationError("The file is not a readable MP4 container.") from exc

    try:
        format_names = {name.strip().lower() for name in (container.format.name or "").split(",")}
        if "mp4" not in format_names:
            LOGGER.warning("Upload rejected: detected container=%s path=%s", container.format.name, path)
            raise UploadValidationError("The uploaded file has an .mp4 name but is not an MP4 container.")
        if not container.streams.video:
            LOGGER.warning("Upload rejected: MP4 has no video stream path=%s", path)
            raise UploadValidationError("The MP4 container does not contain a video stream.")

        stream = container.streams.video[0]
        codec = stream.codec_context.name or "unknown"
        width, height = int(stream.width or 0), int(stream.height or 0)
        fps = float(stream.average_rate or stream.base_rate or stream.guessed_rate or 0)
        duration = float(stream.duration * stream.time_base) if stream.duration is not None else 0.0
        if duration <= 0 and container.duration is not None:
            duration = float(container.duration / av.time_base)
        frame_count = int(stream.frames or 0)
        if frame_count <= 0 and duration > 0 and fps > 0:
            frame_count = int(round(duration * fps))
        if width <= 0 or height <= 0 or fps <= 0 or duration <= 0 or frame_count <= 0:
            LOGGER.warning(
                "Upload rejected: invalid video metadata codec=%s dimensions=%sx%s fps=%s duration=%s frames=%s path=%s",
                codec, width, height, fps, duration, frame_count, path,
            )
            raise UploadValidationError("The MP4 video stream has incomplete or invalid metadata.")
    finally:
        container.close()

    try:
        # ImageIO/PyAV is also used by detector.get_video_frames_generator.
        props = iio.improps(path, plugin="pyav")
        probe = iio.imread(path, index=0, plugin="pyav")
        if probe is None or getattr(probe, "ndim", 0) < 2:
            raise ValueError("decoded first frame has no image dimensions")
        pipeline_frames = props.shape[0] if props.shape else 0
        if not math.isfinite(float(pipeline_frames)) or int(pipeline_frames) <= 0:
            raise ValueError(f"pipeline reported invalid frame count: {pipeline_frames}")
    except Exception as exc:
        error_name = type(exc).__name__.lower()
        LOGGER.warning("Upload rejected: ImageIO/PyAV frame decode failed codec=%s path=%s error=%r", codec, path, exc)
        _ffprobe_diagnostics(path)
        if "decodernotfound" in error_name or "unknowncodec" in error_name:
            raise UploadValidationError(f"The MP4 uses the unsupported video codec '{codec}'.") from exc
        raise UploadValidationError(f"The MP4 container is valid, but its '{codec}' video frames could not be decoded.") from exc

    if duration > max_duration_seconds:
        LOGGER.warning("Upload rejected: duration %.1fs exceeds limit %.1fs path=%s", duration, max_duration_seconds, path)
        raise UploadValidationError(f"Video duration is {duration:.1f}s; the configured limit is {max_duration_seconds:.0f}s.")
    LOGGER.info(
        "Upload validated path=%s container=mp4 codec=%s dimensions=%sx%s fps=%.3f duration=%.3fs frames=%s",
        path, codec, width, height, fps, duration, frame_count,
    )
    _ffprobe_diagnostics(path)
    return VideoMetadata(path, path.name, size_mb, duration, width, height, frame_count, fps)


def extract_research_frames(video_path: Path, face_detector: Any) -> tuple[list[Any], list[int]]:
    """Collect 64 valid faces, restore chronology, then select the GRU-32 frame set."""
    try:
        from temporal.video_loader import load_aligned_frame_sequence

        frames64, ids64 = load_aligned_frame_sequence(
            video_path=str(video_path), face_detector=face_detector, sequence_length=64,
            scale=1.3, target_size=(224, 224),
        )
    except RuntimeError as exc:
        if "usable face frames" in str(exc):
            raise FaceExtractionError("The video does not contain 64 detectable, alignable face frames required by the trained pipeline.") from exc
        raise FaceExtractionError("Face extraction failed for this video.") from exc
    except Exception as exc:
        raise FaceExtractionError("The research face-alignment pipeline could not process this video.") from exc
    frames32, ids32 = frames64[::2], ids64[::2]
    if len(frames32) != 32 or len(ids32) != 32:
        raise FaceExtractionError("Preprocessing did not produce the required 32-frame sequence.")
    if ids32 != sorted(ids32) or len(set(ids32)) != 32:
        raise FaceExtractionError("Preprocessed frames are not strictly chronological and unique.")
    return frames32, ids32
