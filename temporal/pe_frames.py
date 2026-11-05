"""Aligned face crops at exact, known frame IDs.

The CLIP GRU-32 reads every other frame of the 64 stored per video. To give the PE GRU the
same moments, we read those exact frame IDs instead of asking the loader to search again.
The video is decoded once, front to back, and only the wanted frames are detected.
"""
import cv2
import imageio.v3 as iio
import numpy as np
from PIL import Image

from detector import align_face


def load_aligned_frames_at_ids(video_path, face_detector, frame_ids, scale=1.3, target_size=(224, 224)):
    """Return (frames, frame_ids): one aligned RGB PIL face per requested ID, in time order.

    Uses the same detection, largest-face choice and alignment as temporal.video_loader.
    Raises if any requested frame has no usable face, because a gap would misalign the sequence.
    """
    wanted = [int(i) for i in frame_ids]
    if len(set(wanted)) != len(wanted):
        raise ValueError('Duplicate frame IDs requested')
    wanted_set, last = set(wanted), max(wanted)
    found = {}
    for index, frame_rgb in enumerate(iio.imiter(video_path, plugin='pyav')):
        if index in wanted_set:
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            detections, landmarks = face_detector.detect(frame)
            if detections is not None and len(detections) and landmarks is not None:
                areas = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
                aligned, _ = align_face(frame, landmarks[int(np.argmax(areas))],
                                        target_size=target_size, scale=scale)
                found[index] = Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
        if index >= last:
            break
    missing = sorted(wanted_set - set(found))
    if missing:
        raise RuntimeError(f'No usable face at frames {missing[:5]} (of {len(missing)} missing) in {video_path}')
    ordered = sorted(wanted)
    return [found[i] for i in ordered], ordered
