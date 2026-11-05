import cv2
import numpy as np
import torch
from PIL import Image

from detector import get_video_frames_generator, align_face


def load_aligned_frame_sequence(
    video_path,
    face_detector,
    sequence_length=16,
    scale=1.3,
    target_size=(224, 224),
):
    """
    Extract exactly `sequence_length` valid face frames from a video.

    The detector searches across the video for usable frames.
    Successful frames are sorted by their original frame number
    before being returned, preserving chronological order.

    Returns:
        frames: list[PIL.Image]
            Chronologically ordered aligned face images.

        frame_ids: list[int]
            Original frame numbers from the video.
    """

    selected_frames = []

    frame_generator = get_video_frames_generator(
        source_path=video_path,
        mask_path=None,
        num_frames=sequence_length,
        mode="at_least",
    )

    if frame_generator is None:
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    for frame, frame_id, _ in frame_generator:

        # RetinaFace detection
        detections, landmarks = face_detector.detect(frame)

        # Nothing detected
        if (
            detections is None
            or len(detections) == 0
            or landmarks is None
        ):
            continue

        # --------------------------------------------------
        # Choose the largest face
        # --------------------------------------------------

        areas = (
            (detections[:, 2] - detections[:, 0])
            *
            (detections[:, 3] - detections[:, 1])
        )

        largest_face_index = np.argmax(areas)

        selected_landmarks = landmarks[
            largest_face_index
        ]

        # --------------------------------------------------
        # Align face using same alignment code as GenD
        # --------------------------------------------------

        aligned_face, _ = align_face(
            frame,
            selected_landmarks,
            target_size=target_size,
            scale=scale,
        )

        # OpenCV uses BGR.
        # PIL / GenD preprocessing expects RGB.
        aligned_face_rgb = cv2.cvtColor(
            aligned_face,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(
            aligned_face_rgb
        )

        selected_frames.append(
            (frame_id, pil_image)
        )

        # Stop once we successfully found 16 usable faces
        if len(selected_frames) == sequence_length:
            break

    # --------------------------------------------------
    # Check that we actually got 16 usable frames
    # --------------------------------------------------

    if len(selected_frames) < sequence_length:
        raise RuntimeError(
            f"Only found {len(selected_frames)} usable "
            f"face frames in {video_path}. "
            f"Expected {sequence_length}."
        )

    # --------------------------------------------------
    # CRITICAL:
    # Restore chronological order
    # --------------------------------------------------

    selected_frames.sort(
        key=lambda item: item[0]
    )

    frame_ids = [
        frame_id
        for frame_id, _ in selected_frames
    ]

    frames = [
        image
        for _, image in selected_frames
    ]

    return frames, frame_ids


def preprocess_frame_sequence(
    frames,
    preprocess,
):
    """
    Apply GenD preprocessing to a chronological frame sequence.

    Input:
        list of PIL images

    Output:
        Tensor:
        [sequence_length, channels, height, width]
    """

    processed_frames = [
        preprocess(frame)
        for frame in frames
    ]

    return torch.stack(
        processed_frames,
        dim=0,
    )