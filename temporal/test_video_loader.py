from temporal.experiment_config import DATA_ROOT
import os

from detector import prepare_model
from temporal.video_loader import load_aligned_frame_sequence


VIDEO_PATH = str(DATA_ROOT / "original_sequences/youtube/c23/videos/183.mp4")

SEQUENCE_LENGTH = 16


def main():

    # Load RetinaFace
    face_detector = prepare_model(
        det_thres=0.1
    )

    frames, frame_ids = load_aligned_frame_sequence(
        video_path=VIDEO_PATH,
        face_detector=face_detector,
        sequence_length=SEQUENCE_LENGTH,
    )

    print()
    print("Video:", VIDEO_PATH)
    print("Number of frames:", len(frames))
    print("Frame IDs:", frame_ids)
    print("First image size:", frames[0].size)

    # Save temporary images so we can visually inspect them
    output_dir = "tmp/sequence_test"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    for i, (image, frame_id) in enumerate(
        zip(frames, frame_ids)
    ):

        output_path = os.path.join(
            output_dir,
            f"{i:02d}_frame_{frame_id:04d}.png"
        )

        image.save(output_path)

    print()
    print(
        f"Saved test sequence to: {output_dir}"
    )


if __name__ == "__main__":
    main()