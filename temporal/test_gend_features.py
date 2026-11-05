from temporal.experiment_config import DATA_ROOT
import torch

from detector import prepare_model
from src.hf.modeling_gend import GenD

from temporal.video_loader import (
    load_aligned_frame_sequence,
    preprocess_frame_sequence,
)

from temporal.extract_features import extract_gend_features


VIDEO_PATH = str(DATA_ROOT / "original_sequences/youtube/c23/videos/183.mp4")

SEQUENCE_LENGTH = 16

MODEL_NAME = "yermandy/GenD_CLIP_L_14"


def main():

    # --------------------------------------------------
    # 1. Choose device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("PyTorch device:", device)
    print("GPU:", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")

    # --------------------------------------------------
    # 2. RetinaFace
    # --------------------------------------------------

    face_detector = prepare_model(
        det_thres=0.1
    )

    # --------------------------------------------------
    # 3. Extract 16 chronological aligned faces
    # --------------------------------------------------

    frames, frame_ids = load_aligned_frame_sequence(
        video_path=VIDEO_PATH,
        face_detector=face_detector,
        sequence_length=SEQUENCE_LENGTH,
    )

    print()
    print("Selected frame IDs:")
    print(frame_ids)

    # --------------------------------------------------
    # 4. Load pretrained GenD
    # --------------------------------------------------

    print()
    print("Loading GenD...")

    gend_model = GenD.from_pretrained(
        MODEL_NAME
    )

    gend_model = gend_model.to(device)
    gend_model.eval()

    # --------------------------------------------------
    # 5. Apply GenD / CLIP preprocessing
    # --------------------------------------------------

    frame_batch = preprocess_frame_sequence(
        frames,
        gend_model.feature_extractor.preprocess,
    )

    frame_batch = frame_batch.to(device)

    print()
    print(
        "Frame tensor shape:",
        frame_batch.shape
    )

    # --------------------------------------------------
    # 6. Extract GenD features
    # --------------------------------------------------

    features = extract_gend_features(
        gend_model,
        frame_batch,
    )

    print()
    print(
        "GenD feature shape:",
        features.shape
    )

    print(
        "Feature dimension:",
        features.shape[-1]
    )

    print()
    print("SUCCESS: 16-frame GenD feature sequence created.")


if __name__ == "__main__":
    main()
    