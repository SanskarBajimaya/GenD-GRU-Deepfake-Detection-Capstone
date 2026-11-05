from __future__ import annotations

import sys
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

FRONTEND = Path(__file__).resolve().parents[1]
ROOT = FRONTEND.parent
for path in (str(FRONTEND),str(ROOT)):
    if path not in sys.path: sys.path.insert(0,path)

from model_loader import MAX_RETINAFACE_BYTES, MODEL_LABELS, ModelManager, RuntimeDependencyError
from predictor import AnalysisError, AnalysisReport, Prediction, PredictionService
from preprocessing import UploadValidationError, inspect_video


class UploadValidationTests(unittest.TestCase):
    def test_missing_upload_is_rejected(self):
        with self.assertRaisesRegex(UploadValidationError,"No video"): inspect_video(None,10,10)

    def test_non_mp4_is_rejected_before_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"sample.txt"; path.write_text("not video",encoding="utf-8")
            with self.assertRaisesRegex(UploadValidationError,"Only MP4"): inspect_video(path,10,10)

    def test_oversized_mp4_is_rejected_before_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"sample.mp4"; path.write_bytes(b"0"*2048)
            with self.assertRaisesRegex(UploadValidationError,"limit"): inspect_video(path,0,10)

    def test_invalid_mp4_container_has_specific_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"sample.mp4"; path.write_bytes(b"this is not an mp4 container")
            with self.assertRaisesRegex(UploadValidationError,"readable MP4 container"):
                inspect_video(path,10,10)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for the integration fixture")
    def test_h264_aac_mp4_passes_the_pipeline_decoder(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"h264-aac.mp4"
            subprocess.run([
                shutil.which("ffmpeg"), "-v", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=24",
                "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000",
                "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
            ], check=True, timeout=30)
            item=inspect_video({"path":str(path)},10,10)
            self.assertEqual((item.width,item.height),(320,180))
            self.assertEqual(item.frame_count,24)
            self.assertAlmostEqual(item.fps,24.0)


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.manager=ModelManager(ROOT); self.service=PredictionService(self.manager)

    def test_single_model_plan_does_not_invent_comparisons(self):
        availability={key:(True,"ready") for key in MODEL_LABELS}
        self.assertEqual(self.service._plan("gend_clip",False,availability),["gend_clip"])

    def test_compare_all_omits_unavailable_models(self):
        available={"gend_clip","gend_pe","head_ensemble"}
        state={key:(key in available,"state") for key in MODEL_LABELS}
        self.assertEqual(self.service._plan("gend_clip",True,state),["gend_clip","gend_pe","head_ensemble"])

    def test_unavailable_selected_model_has_clear_error(self):
        state={key:(key!="clip_gru","missing") for key in MODEL_LABELS}
        with self.assertRaisesRegex(AnalysisError,"unavailable"): self.service._plan("clip_gru",False,state)

    def test_required_backbones_for_ensembles(self):
        self.assertEqual(self.service._required_backbones(["head_ensemble"]),["clip","pe"])
        self.assertEqual(self.service._required_backbones(["clip_gru"]),["clip"])

    def test_all_five_seed_paths_are_required(self):
        self.assertEqual([p.name for p in self.manager.checkpoint_paths("clip")],[f"gru32_seed{i}_best.pt" for i in range(5)])
        self.assertEqual([p.name for p in self.manager.checkpoint_paths("pe")],[f"gru32_pe_seed{i}_best.pt" for i in range(5)])

    def test_face_detector_download_is_size_bounded(self):
        class OversizedResponse:
            headers = {"Content-Length": str(MAX_RETINAFACE_BYTES + 1)}
            def __enter__(self): return self
            def __exit__(self, *args): return False

        with tempfile.TemporaryDirectory() as directory:
            manager = ModelManager(Path(directory))
            with patch("model_loader.urlopen", return_value=OversizedResponse()):
                with self.assertRaisesRegex(RuntimeDependencyError, "safety limit"):
                    manager._ensure_face_detector_weight()
            self.assertFalse((Path(directory)/"weights"/"models"/"buffalo_l"/"det_10g.onnx.partial").exists())


class UiTests(unittest.TestCase):
    def test_result_uses_fake_probability_direction(self):
        import app
        pred=Prediction("gend_clip","GenD-CLIP",.8742)
        report=AnalysisReport(pred,(pred,),1.2,32,"CPU",("test note",))
        rendered=app._result_html(report)
        self.assertIn("FAKE",rendered); self.assertIn("87.42%",rendered); self.assertIn("Model-estimated fake probability",rendered)

    def test_demo_constructs_without_loading_ml_models(self):
        import app
        self.assertIsNotNone(app.demo); self.assertIn("GenD",app.APP_TITLE)

    def test_video_upload_preserves_mp4_and_audio(self):
        import app
        component=next(item for item in app.demo.config["components"] if item["type"]=="video")
        self.assertTrue(component["props"]["include_audio"])
        self.assertNotIn("format",component["props"])


if __name__ == "__main__": unittest.main()
