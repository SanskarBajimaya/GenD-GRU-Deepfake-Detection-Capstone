from pathlib import Path

import torch

from detector import prepare_model
from src.hf.modeling_gend import GenD

from temporal.video_loader import (
    load_aligned_frame_sequence,
    preprocess_frame_sequence,
)

from temporal.extract_features import extract_gend_features
from temporal.experiment_config import DATA_ROOT


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEQUENCE_LENGTH = 16

MODEL_NAME = "yermandy/GenD_CLIP_L_14"

REAL_VIDEO_DIR = DATA_ROOT / "original_sequences/youtube/c23/videos"

FAKE_VIDEO_DIR = DATA_ROOT / "manipulated_sequences/Deepfakes/c23/videos"

OUTPUT_ROOT = Path(
    "temporal/features"
)


def process_video(
    video_path,
    output_path,
    label,
    face_detector,
    gend_model,
    device,
):

    print()
    print("=" * 60)
    print("Processing:", video_path)
    print("Label:", "REAL" if label == 0 else "FAKE")

    # --------------------------------------------------
    # 1. Get 16 chronological aligned faces
    # --------------------------------------------------

    frames, frame_ids = load_aligned_frame_sequence(
        video_path=str(video_path),
        face_detector=face_detector,
        sequence_length=SEQUENCE_LENGTH,
    )

    print("Frame IDs:", frame_ids)

    # --------------------------------------------------
    # 2. GenD preprocessing
    # --------------------------------------------------

    frame_batch = preprocess_frame_sequence(
        frames,
        gend_model.feature_extractor.preprocess,
    )

    frame_batch = frame_batch.to(device)

    # --------------------------------------------------
    # 3. GenD feature extraction
    # --------------------------------------------------

    features = extract_gend_features(
        gend_model,
        frame_batch,
    )

    print("Feature shape:", features.shape)

    # --------------------------------------------------
    # 4. Save features
    # --------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        features.cpu(),
        output_path,
    )

    print("Saved:", output_path)


def main():

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("PyTorch device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------
    # RetinaFace
    # --------------------------------------------------

    print()
    print("Loading RetinaFace...")

    face_detector = prepare_model(
        det_thres=0.1
    )

    # --------------------------------------------------
    # GenD
    # --------------------------------------------------

    print()
    print("Loading GenD...")

    gend_model = GenD.from_pretrained(
        MODEL_NAME
    )

    gend_model = gend_model.to(device)
    gend_model.eval()

    # --------------------------------------------------
    # Dataset groups
    # --------------------------------------------------

    datasets = [
        (
            REAL_VIDEO_DIR,
            OUTPUT_ROOT / "real",
            0,
        ),
        (
            FAKE_VIDEO_DIR,
            OUTPUT_ROOT / "fake",
            1,
        ),
    ]

    total_processed = 0

    # --------------------------------------------------
    # Process videos
    # --------------------------------------------------

    for video_dir, output_dir, label in datasets:

        video_files = sorted(
            video_dir.glob("*.mp4")
        )

        print()
        print(
            f"Found {len(video_files)} videos in "
            f"{video_dir}"
        )

        for video_path in video_files:

            output_path = (
                output_dir
                / f"{video_path.stem}.pt"
            )

            # Don't repeat completed work
            if output_path.exists():
                print(
                    "Skipping existing:",
                    output_path
                )
                continue

            try:

                process_video(
                    video_path=video_path,
                    output_path=output_path,
                    label=label,
                    face_detector=face_detector,
                    gend_model=gend_model,
                    device=device,
                )

                total_processed += 1

            except Exception as error:

                print()
                print(
                    "ERROR processing:",
                    video_path
                )

                print(error)

    print()
    print("=" * 60)
    print(
        "Finished."
    )

    print(
        "New videos processed:",
        total_processed
    )


if __name__ == "__main__":
    main()
