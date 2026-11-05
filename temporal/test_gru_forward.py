from temporal.experiment_config import DATA_ROOT
import torch

from detector import prepare_model
from src.hf.modeling_gend import GenD

from temporal.video_loader import (
    load_aligned_frame_sequence,
    preprocess_frame_sequence,
)

from temporal.extract_features import extract_gend_features
from temporal.model_a import TemporalGRU


VIDEO_PATH = str(DATA_ROOT / "original_sequences/youtube/c23/videos/183.mp4")

SEQUENCE_LENGTH = 16
MODEL_NAME = "yermandy/GenD_CLIP_L_14"


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------
    # 1. RetinaFace
    # --------------------------------------------------

    face_detector = prepare_model(
        det_thres=0.1
    )

    # --------------------------------------------------
    # 2. Load 16 chronological aligned faces
    # --------------------------------------------------

    frames, frame_ids = load_aligned_frame_sequence(
        video_path=VIDEO_PATH,
        face_detector=face_detector,
        sequence_length=SEQUENCE_LENGTH,
    )

    print("Frame IDs:", frame_ids)

    # --------------------------------------------------
    # 3. Load GenD
    # --------------------------------------------------

    gend_model = GenD.from_pretrained(
        MODEL_NAME
    )

    gend_model = gend_model.to(device)
    gend_model.eval()

    # --------------------------------------------------
    # 4. GenD preprocessing
    # --------------------------------------------------

    frame_batch = preprocess_frame_sequence(
        frames,
        gend_model.feature_extractor.preprocess,
    ).to(device)

    print(
        "Frame tensor shape:",
        frame_batch.shape
    )

    # --------------------------------------------------
    # 5. GenD features
    # --------------------------------------------------

    features = extract_gend_features(
        gend_model,
        frame_batch,
    )

    print(
        "Feature sequence shape:",
        features.shape
    )

    # --------------------------------------------------
    # 6. Add batch dimension
    # --------------------------------------------------

    features = features.unsqueeze(0)

    print(
        "GRU input shape:",
        features.shape
    )

    # --------------------------------------------------
    # 7. Create GRU
    # --------------------------------------------------

    feature_dim = features.shape[-1]

    gru_model = TemporalGRU(
        feature_dim=feature_dim,
        hidden_dim=128,
    ).to(device)

    gru_model.eval()

    # --------------------------------------------------
    # 8. Forward pass
    # --------------------------------------------------

    with torch.no_grad():
        logit = gru_model(features)

    print(
        "GRU output shape:",
        logit.shape
    )

    print(
        "Raw logit:",
        logit
    )

    probability = torch.sigmoid(logit)

    print(
        "Fake probability:",
        probability
    )

    print()
    print(
        "SUCCESS: complete GenD -> GRU forward pass works."
    )


if __name__ == "__main__":
    main()