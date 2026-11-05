"""Evaluation only: original GenD probability averaging versus trained Model A.

Pretrained GenD was trained on FaceForensics++ c23. This four-video test is
an in-domain development comparison, not evidence of cross-dataset
 generalization or statistically significant improvement.
"""
from pathlib import Path

import torch
import torch.nn.functional as F

from detector import prepare_model
from src.hf.modeling_gend import GenD
from temporal.dataset_split import create_paired_splits
from temporal.feature_dataset import TemporalFeatureDataset
from temporal.model_a import TemporalGRU
from temporal.video_loader import load_aligned_frame_sequence, preprocess_frame_sequence
from temporal.experiment_config import DATA_ROOT

ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_LENGTH = 16
MODEL_NAME = "yermandy/GenD_CLIP_L_14"
EXPECTED_FRAME_IDS = {
    "866": [0, 31, 63, 95, 127, 158, 190, 222, 254, 285, 317, 349, 381, 444, 476, 508],
    "878": [0, 46, 92, 138, 184, 230, 276, 322, 368, 459, 505, 551, 597, 643, 689, 735],
    "866_878": [0, 31, 63, 95, 127, 158, 190, 222, 254, 285, 317, 349, 381, 444, 476, 508],
    "878_866": [0, 46, 92, 138, 184, 230, 276, 322, 368, 459, 505, 551, 597, 643, 689, 735],
}
EXPECTED_GRU_PROBS = {"866": 0.0179, "878": 0.0337, "866_878": 0.9492, "878_866": 0.8970}
# Absolute probability tolerance permits floating-point variation, not a changed experiment.
GRU_PROB_TOLERANCE = 0.001


def class_name(value):
    return "FAKE" if value >= 0.5 else "REAL"


def print_results(rows):
    print("\n" + "=" * 76)
    print("GenD vs GenD+GRU")
    print("=" * 76)
    print(f"{'Video':<12} {'True':<6} {'GenD Prob':>10} {'GenD Pred':>10} {'GRU Prob':>10} {'GRU Pred':>10}")
    print("-" * 76)
    for row in rows:
        print(f"{row['video']:<12} {class_name(row['label']):<6} "
              f"{row['gend']:>10.4f} {class_name(row['gend']):>10} "
              f"{row['gru']:>10.4f} {class_name(row['gru']):>10}")
    print("-" * 76)
    labels = torch.tensor([r["label"] for r in rows], dtype=torch.float64)
    for key, title in (("gend", "GenD"), ("gru", "GRU")):
        probs = torch.tensor([r[key] for r in rows], dtype=torch.float64)
        correct = int(((probs >= 0.5) == labels.bool()).sum())
        # BCE on the final video probabilities, not frame logits.
        loss = F.binary_cross_entropy(probs, labels).item()
        print(f"{title} Accuracy: {correct}/{len(rows)} = {100 * correct / len(rows):.1f}%")
        print(f"{title} video BCE/log loss: {loss:.6f}")
    print("=" * 76)
    print("Scientific note: pretrained GenD was trained on FaceForensics++ c23.")
    print("These four videos provide an in-domain development baseline only.")
    print("GenD pretraining overlap with these videos has not been ruled out.")
    print("This does not prove cross-dataset generalization or statistical significance.")


@torch.inference_mode()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    dataset = TemporalFeatureDataset(
        feature_root=ROOT / "temporal/features",
        expected_sequence_length=SEQUENCE_LENGTH,
        expected_feature_dim=1024,
    )
    _, _, test_dataset = create_paired_splits(dataset)
    samples = [test_dataset[i] for i in range(len(test_dataset))]
    names = [sample["video"] for sample in samples]
    if len(names) != 4 or set(names) != set(EXPECTED_FRAME_IDS):
        raise RuntimeError(f"Held-out split differs from expected four videos: {names}")
    samples.sort(key=lambda sample: list(EXPECTED_FRAME_IDS).index(sample["video"]))

    checkpoint = torch.load(ROOT / "temporal/models/best_gru.pt", map_location=device, weights_only=True)
    if checkpoint["sequence_length"] != SEQUENCE_LENGTH or checkpoint["feature_dim"] != 1024:
        raise RuntimeError("GRU checkpoint does not describe [16, 1024] feature sequences.")
    gru = TemporalGRU(feature_dim=checkpoint["feature_dim"], hidden_dim=checkpoint["hidden_dim"])
    gru.load_state_dict(checkpoint["model_state_dict"], strict=True)
    gru = gru.to(device).eval()
    gru.requires_grad_(False)

    # Use the unmodified saved tensors from the existing GRU experiment.
    saved_batch = torch.stack([s["features"] for s in samples]).to(device)
    gru_probs = torch.sigmoid(gru(saved_batch)).cpu().tolist()
    for sample, probability in zip(samples, gru_probs):
        expected = EXPECTED_GRU_PROBS[sample["video"]]
        print(f"GRU reproduction {sample['video']}: {probability:.6f} (expected ~{expected:.4f})")
        if abs(probability - expected) > GRU_PROB_TOLERANCE:
            raise RuntimeError(f"GRU discrepancy for {sample['video']}: {probability:.6f} vs {expected:.4f}; comparison stopped.")

    face_detector = prepare_model(det_thres=0.1)
    # Verify ALL four sequences before calculating any baseline results.
    sequences = []
    for sample in samples:
        video = sample["video"]
        label = int(sample["label"].item())
        expected_label = int("_" in video)
        if label != expected_label:
            raise RuntimeError(f"Unexpected label for {video}: {label}")
        subdir = "manipulated_sequences/Deepfakes" if label else "original_sequences/youtube"
        video_path = DATA_ROOT / subdir / "c23/videos" / f"{video}.mp4"
        frames, frame_ids = load_aligned_frame_sequence(
            video_path=str(video_path), face_detector=face_detector, sequence_length=SEQUENCE_LENGTH,
        )
        print(f"Selected frame IDs for {video}: {frame_ids}")
        if len(frames) != SEQUENCE_LENGTH or frame_ids != EXPECTED_FRAME_IDS[video]:
            raise RuntimeError(f"Frame discrepancy for {video}: expected {EXPECTED_FRAME_IDS[video]}, got {frame_ids}; comparison stopped.")
        sequences.append((frames, frame_ids))

    gend, loading_info = GenD.from_pretrained(MODEL_NAME, output_loading_info=True)
    # A missing head would create a random baseline; reject incomplete loads.
    if any(loading_info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"GenD pretrained loading discrepancy: {loading_info}")
    gend = gend.to(device).eval()
    gend.requires_grad_(False)
    print("GenD head:", gend.model)
    print("GenD output: raw Tensor [16, 2], class 0 REAL, class 1 FAKE.")
    print("Aggregation: softmax(logits, dim=1)[:, 1].mean(); threshold >= 0.5 is FAKE.")

    rows = []
    for sample, (frames, frame_ids), gru_probability in zip(samples, sequences, gru_probs):
        frame_batch = preprocess_frame_sequence(frames, gend.feature_extractor.preprocess).to(device)
        if tuple(frame_batch.shape) != (16, 3, 224, 224):
            raise RuntimeError(f"Unexpected frame tensor shape: {tuple(frame_batch.shape)}")
        logits = gend(frame_batch)  # Original pretrained head, via the original HF forward API.
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (16, 2):
            raise RuntimeError("GenD did not return the inspected [16, 2] logits tensor.")
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"Nonfinite GenD logits for {sample['video']}")
        frame_probs = logits.softmax(dim=1)[:, 1]
        video_probability = frame_probs.mean().item()
        print(f"\n{sample['video']} GenD frame probabilities:")
        for frame_id, probability in zip(frame_ids, frame_probs.cpu().tolist()):
            print(f"frame {frame_id}: {probability:.6f}")
        print(f"Average: {video_probability:.6f}")
        rows.append({"video": sample["video"], "label": int(sample["label"].item()),
                     "gend": video_probability, "gru": gru_probability})
    print_results(rows)


if __name__ == "__main__":
    main()
