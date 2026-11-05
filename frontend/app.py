"""Premium local Gradio dashboard for genuine GenD/GRU video inference."""

from __future__ import annotations

import html
import logging
import os
import sys
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FRONTEND_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

import gradio as gr
import pandas as pd

logging.basicConfig(
    level=getattr(logging, os.getenv("GEN_D_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from model_loader import ModelManager
from predictor import MODEL_LABELS, AnalysisError, AnalysisReport, PredictionService
from preprocessing import UploadValidationError, inspect_video

APP_TITLE = "GenD Deepfake Intelligence"
MAX_UPLOAD_MB = int(os.getenv("GEN_D_MAX_UPLOAD_MB", "500"))
MAX_DURATION_SECONDS = float(os.getenv("GEN_D_MAX_DURATION_SECONDS", "300"))
CSS_PATH = FRONTEND_DIR / "style.css"

MANAGER = ModelManager(PROJECT_ROOT)
SERVICE = PredictionService(MANAGER)


def _icon(name: str) -> str:
    icons = {
        "shield": """<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3 4.5 6v5.2c0 4.8 3.1 8.3 7.5 9.8 4.4-1.5 7.5-5 7.5-9.8V6L12 3Z"/><path d="m8.8 12 2 2 4.5-4.5"/></svg>""",
        "scan": """<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3"/><circle cx="12" cy="12" r="3"/></svg>""",
        "pulse": """<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12h4l2-6 4 12 2-6h6"/></svg>""",
    }
    return icons[name]


def _nav_html() -> str:
    return f"""
    <header class="top-nav">
      <div class="brand-lockup"><div class="brand-mark">{_icon('shield')}</div>
        <div><div class="brand-name">GenD</div><div class="brand-product">Deepfake Intelligence</div></div></div>
      <div class="nav-center">Video Authenticity Analysis</div>
      <div class="nav-status"><span class="status-dot"></span> Research Prototype
        <span class="device-pill">{html.escape(MANAGER.device_label)}</span></div>
    </header>"""


def _hero_html() -> str:
    return f"""
    <section class="hero-shell"><div class="hero-orb orb-one"></div><div class="hero-orb orb-two"></div><div class="hero-grid"></div>
      <div class="hero-content"><div class="eyebrow"><span>{_icon('scan')}</span> MULTI-BACKBONE FORENSIC ANALYSIS</div>
        <h1>Detect the <span>Difference.</span></h1>
        <p>Advanced deepfake analysis powered by GenD visual encoders and GRU temporal intelligence.</p>
        <div class="hero-facts"><span>CLIP ViT-L/14</span><i></i><span>Perception Encoder</span><i></i><span>32-frame temporal reasoning</span></div></div>
      <div class="hero-visual"><div class="scanner-ring ring-a"></div><div class="scanner-ring ring-b"></div>
        <div class="scanner-core">{_icon('pulse')}<strong>FORENSIC</strong><small>PIPELINE</small></div></div>
    </section>"""


EMPTY_RESULT = """
<div class="empty-result"><div class="empty-visual"><div class="empty-ring"></div><div class="empty-scan">◇</div></div>
  <h3>Awaiting analysis</h3><p>Upload a valid MP4 video, choose a detector, and begin forensic processing.</p>
  <div class="empty-steps"><span>01 Face alignment</span><span>02 Feature extraction</span><span>03 Authenticity signal</span></div></div>
"""


def _availability_html() -> str:
    rows = []
    for key, label in MODEL_LABELS.items():
        available, reason = MANAGER.availability()[key]
        state, status = ("ready", "Available") if available else ("blocked", "Weights required")
        rows.append(f'<div class="availability-row"><span class="availability-state {state}"></span><div><strong>{html.escape(label)}</strong><small>{html.escape(reason)}</small></div><b>{status}</b></div>')
    return '<div class="availability-list">' + "".join(rows) + "</div>"


def _upload_info_html(video_path: str | None) -> str:
    if not video_path:
        return '<div class="file-meta empty"><span>No video selected</span><small>MP4 · configured size and duration limits apply</small></div>'
    try:
        item = inspect_video(video_path, MAX_UPLOAD_MB, MAX_DURATION_SECONDS)
    except UploadValidationError as exc:
        return f'<div class="file-meta error"><span>Upload rejected</span><small>{html.escape(str(exc))}</small></div>'
    return f"""<div class="file-meta valid"><div><span>{html.escape(item.safe_name)}</span><small>Validated MP4 container</small></div>
      <dl><div><dt>Size</dt><dd>{item.size_mb:.1f} MB</dd></div><div><dt>Duration</dt><dd>{item.duration_seconds:.1f}s</dd></div>
      <div><dt>Resolution</dt><dd>{item.width} × {item.height}</dd></div><div><dt>Frames</dt><dd>{item.frame_count:,}</dd></div></dl></div>"""


def _result_html(report: AnalysisReport) -> str:
    selected = report.selected
    fake_pct, real_pct = selected.fake_probability * 100, (1 - selected.fake_probability) * 100
    classification = "FAKE" if selected.fake_probability >= 0.5 else "REAL"
    tone = classification.lower()
    notes = "".join(f"<li>{html.escape(note)}</li>" for note in report.notes)
    return f"""<div class="result-stack {tone}"><div class="verdict-row"><div><div class="metric-kicker">ANALYTICAL VERDICT</div>
      <div class="verdict-badge {tone}"><span></span>{classification}</div></div><div class="score-dial" style="--score:{fake_pct:.2f}%">
      <div><strong>{fake_pct:.2f}%</strong><small>FAKE SIGNAL</small></div></div></div>
      <div class="probability-block"><div class="probability-head"><span>Model-estimated fake probability</span><strong>{fake_pct:.2f}%</strong></div>
      <div class="probability-track"><i class="real-segment" style="width:{real_pct:.2f}%"></i><i class="fake-segment" style="width:{fake_pct:.2f}%"></i></div>
      <div class="probability-labels"><span>REAL {real_pct:.2f}%</span><span>FAKE {fake_pct:.2f}%</span></div></div>
      <div class="result-metrics"><div><small>MODEL</small><strong>{html.escape(selected.label)}</strong></div><div><small>PROCESSING</small><strong>{report.processing_seconds:.1f}s</strong></div>
      <div><small>FRAMES USED</small><strong>{report.frames_used}</strong></div><div><small>COMPUTE</small><strong>{html.escape(report.device)}</strong></div></div>
      <div class="technical-notes"><strong>Technical notes</strong><ul>{notes}</ul></div></div>"""


def _status_html(message: str, tone: str = "neutral") -> str:
    return f'<div class="run-status {tone}"><span></span>{html.escape(message)}</div>'


def _error_html(title: str, message: str) -> str:
    return f'<div class="analysis-error"><div class="error-icon">!</div><div><h3>{html.escape(title)}</h3><p>{html.escape(message)}</p><small>No prediction was produced.</small></div></div>'


def _chart_frame(report: AnalysisReport | None = None) -> pd.DataFrame:
    if report is None:
        return pd.DataFrame({"Model": [], "Fake probability (%)": [], "Prediction": []})
    return pd.DataFrame({"Model": [p.label for p in report.predictions], "Fake probability (%)": [round(p.fake_probability * 100, 2) for p in report.predictions], "Prediction": ["FAKE" if p.fake_probability >= .5 else "REAL" for p in report.predictions]})


def _run_analysis(video_path: str | None, model_key: str, compare_all: bool, progress=gr.Progress()):
    if not video_path:
        return EMPTY_RESULT, _chart_frame(), _status_html("Select an MP4 video before analysis.", "error")
    try:
        progress(.02, desc="Validating video container")
        metadata = inspect_video(video_path, MAX_UPLOAD_MB, MAX_DURATION_SECONDS)
        def update(fraction: float, message: str) -> None:
            progress(min(max(fraction, .08), .98), desc=message)
        report = SERVICE.analyze(metadata.path, model_key, compare_all, metadata, update)
        progress(1, desc="Analysis complete")
        count = len(report.predictions)
        return _result_html(report), _chart_frame(report), _status_html(f"Analysis complete · {count} genuine model result{'s' if count != 1 else ''}", "success")
    except (UploadValidationError, AnalysisError) as exc:
        return _error_html("Analysis unavailable", str(exc)), _chart_frame(), _status_html(str(exc), "error")
    except Exception as exc:
        return _error_html("Inference failed safely", "The inference service encountered an unexpected local error."), _chart_frame(), _status_html(type(exc).__name__, "error")


def run_selected(video_path: str | None, model_key: str, progress=gr.Progress()):
    return _run_analysis(video_path, model_key, False, progress)


def run_comparison(video_path: str | None, model_key: str, progress=gr.Progress()):
    return _run_analysis(video_path, model_key, True, progress)


def _model_cards_html() -> str:
    cards = [
        ("01", "GenD-CLIP", "Spatial foundation", "CLIP ViT-L/14 encodes aligned faces; the released GenD head scores every frame and the video score is their mean."),
        ("02", "GenD-PE", "Generalized visual signal", "Perception Encoder Core-L supplies an independent 1,024-dimensional representation with its released GenD classification head."),
        ("03", "CLIP-GRU-32", "Temporal sequence", "Five validated GRU seeds read the same 32 chronological CLIP embeddings and their video probabilities are averaged."),
        ("04", "PE-GRU-32", "Temporal sequence", "The matching five-seed GRU configuration operates on PE embeddings extracted from exactly corresponding aligned frames."),
        ("05", "Original-head ensemble", "Score fusion", "Mean probability fusion combines genuine GenD-CLIP and GenD-PE video scores without mixing benchmark metrics."),
        ("06", "Temporal GRU ensemble", "Score fusion", "Mean probability fusion combines the five-seed CLIP-GRU and PE-GRU video-level outputs."),
    ]
    return '<div class="model-card-grid">' + "".join(f'<article class="model-card"><div class="model-index">{idx}</div><small>{subtitle}</small><h3>{title}</h3><p>{body}</p><span>1,024-D SIGNAL</span></article>' for idx, title, subtitle, body in cards) + "</div>"


def _benchmark_html() -> str:
    values = [("GenD-CLIP",95.41),("GenD-PE",95.90),("Original ensemble",97.12),("CLIP-GRU-32",94.63),("PE-GRU-32",94.38),("GRU ensemble",96.40)]
    bars = "".join(f'<div class="benchmark-row"><span>{name}</span><div><i style="width:{score}%"></i></div><strong>{score:.2f}%</strong></div>' for name, score in values)
    return f'<section class="benchmark-card"><div class="section-heading compact"><div><span>RESEARCH CONTEXT</span><h2>Celeb-DF v2 benchmark</h2></div><p>Historical dataset-level AUROC · 518 videos · not an uploaded-video probability</p></div><div class="benchmark-bars">{bars}</div></section>'


def build_demo() -> gr.Blocks:
    availability = MANAGER.availability()
    selectable = [key for key in MODEL_LABELS if availability[key][0]]
    default_model = selectable[0] if selectable else "gend_clip"
    with gr.Blocks(title=APP_TITLE, fill_width=True, delete_cache=(3600, 3600)) as demo:
        gr.HTML(_nav_html()); gr.HTML(_hero_html())
        with gr.Row(elem_classes=["workspace-grid"], equal_height=False):
            with gr.Column(scale=6, elem_classes=["glass-panel", "analysis-panel"]):
                gr.HTML('<div class="panel-heading"><div><span>01 · INPUT</span><h2>Video analysis</h2></div><div class="secure-chip">LOCAL PROCESSING</div></div>')
                # Keep an already-valid MP4 byte-for-byte. Gradio otherwise invokes
                # FFmpeg solely to strip audio and hands callbacks a muted_*.mp4 copy.
                video = gr.Video(label="Drop an MP4 video here", sources=["upload"], format=None, include_audio=True, buttons=[], elem_id="video-upload", elem_classes=["video-uploader"])
                upload_info = gr.HTML(_upload_info_html(None), elem_id="upload-info")
                with gr.Group(elem_classes=["control-group"]):
                    model = gr.Dropdown(choices=[(MODEL_LABELS[key], key) for key in selectable], value=default_model, label="Detection configuration", info="Only configurations with all required local weights are selectable.", interactive=bool(selectable), elem_id="model-selector")
                    with gr.Row(elem_classes=["action-row"]):
                        analyze = gr.Button("Analyze video", variant="primary", elem_id="analyze-btn", scale=3)
                        compare = gr.Button("Compare available", variant="secondary", elem_id="compare-btn", scale=2)
                gr.HTML(_availability_html(), elem_id="availability-card")
            with gr.Column(scale=7, elem_classes=["glass-panel", "results-panel"]):
                gr.HTML('<div class="panel-heading"><div><span>02 · OUTPUT</span><h2>Detection results</h2></div><div class="live-chip"><i></i> MODEL SIGNAL</div></div>')
                status = gr.HTML(_status_html("Ready for a local video."), elem_id="run-status")
                result = gr.HTML(EMPTY_RESULT, elem_id="result-output")
        gr.HTML('<div class="section-heading"><div><span>MODEL COMPARISON</span><h2>Uploaded-video probability signals</h2></div><p>Only models actually executed for this upload appear below.</p></div>')
        chart = gr.BarPlot(value=_chart_frame(), x="Model", y="Fake probability (%)", color="Prediction", color_map={"REAL":"#34D399","FAKE":"#FB7185"}, y_lim=[0,100], y_title="Model-estimated fake probability (%)", x_title=None, tooltip=["Model","Fake probability (%)","Prediction"], height=360, buttons=["fullscreen","export"], show_label=False, elem_id="comparison-chart")
        gr.HTML('<div class="section-heading"><div><span>DETECTION ARCHITECTURE</span><h2>Six complementary configurations</h2></div><p>Released visual heads and capstone temporal extensions.</p></div>')
        gr.HTML(_model_cards_html()); gr.HTML(_benchmark_html())
        gr.HTML('<footer class="dashboard-footer"><div><strong>GenD + GRU Deepfake Detection</strong><span>University capstone research prototype</span></div><p>Model predictions are experimental and should not be considered definitive proof of video authenticity.</p></footer>')
        video.change(fn=lambda path:_upload_info_html(path), inputs=video, outputs=upload_info, show_progress="hidden", api_visibility="private")
        analyze.click(fn=run_selected, inputs=[video,model], outputs=[result,chart,status], concurrency_id="inference", concurrency_limit=1, scroll_to_output=True, api_visibility="private", show_progress="minimal")
        compare.click(fn=run_comparison, inputs=[video,model], outputs=[result,chart,status], concurrency_id="inference", concurrency_limit=1, scroll_to_output=True, api_visibility="private", show_progress="minimal")
    return demo.queue(max_size=4, default_concurrency_limit=1, api_open=False)


demo = build_demo()

if __name__ == "__main__":
    blocked = [PROJECT_ROOT/"data",PROJECT_ROOT/"datasets",PROJECT_ROOT/"weights",PROJECT_ROOT/"runs",PROJECT_ROOT/"temporal"/"models",PROJECT_ROOT/"temporal"/"models_pe"]
    demo.launch(server_name=os.getenv("GEN_D_HOST","127.0.0.1"), server_port=int(os.getenv("GEN_D_PORT","7860")), share=False, show_error=False, max_file_size=f"{MAX_UPLOAD_MB}mb", blocked_paths=[str(p) for p in blocked], enable_monitoring=False, footer_links=[], theme=gr.themes.Base(primary_hue="sky",neutral_hue="slate"), css_paths=CSS_PATH)
