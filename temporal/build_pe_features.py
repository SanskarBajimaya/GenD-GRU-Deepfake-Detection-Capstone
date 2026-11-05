"""Frozen GenD-PE feature extraction for the paired FaceForensics++ videos.

Mirrors temporal/build_master_features.py (CLIP) without modifying it. Writes 32-frame files to
temporal/features_pe32/ so the CLIP masters in temporal/features_64/ are never touched.
The 32 frames are exactly the ones the selected CLIP GRU-32 reads (every other stored CLIP frame).
Default is a small smoke test that also reports timing. Use --all for every video.
"""
import argparse
import re
import time
import torch
from temporal.build_manifest import load_definitions, save_json
from temporal.build_master_features import encode_frames
from temporal.experiment_config import ROOT, FEATURE_ROOT, RESULT_ROOT
from temporal.master_dataset import master_path

PE_MODEL_NAME = 'yermandy/GenD_PE_L'
PE_FEATURE_ROOT = ROOT / 'temporal/features_pe32'
SEQUENCE_LENGTH = 32


def check_frame_alignment(video_id, pe_frame_ids, clip_frame_ids):
    """Decide whether PE and CLIP used the same frames for this video.

    pe_frame_ids / clip_frame_ids: lists of 32 ints (CLIP's is None if no CLIP file exists).
    Return True to accept the video, False to count it as a failure.

    Strict policy: the two GRUs must see exactly the same moments of each video, so any
    difference is a failure. A video with no CLIP file (real/029 is missing) is accepted,
    since there is nothing to compare against.
    """
    if clip_frame_ids is None:
        print(f'  note: no CLIP features for {video_id}; alignment not checked', flush=True)
        return True
    if list(pe_frame_ids) == list(clip_frame_ids):
        return True
    shared = len(set(pe_frame_ids) & set(clip_frame_ids))
    print(f'  frame mismatch for {video_id}: only {shared}/{len(pe_frame_ids)} IDs shared', flush=True)
    return False


def validate_pe(value, record, feature_dim):
    for field in ('video_id', 'label', 'group_id', 'frame_ids', 'features', 'gend_logits', 'model_name'):
        if field not in value:
            raise ValueError(f'Missing {field}')
    if value['model_name'] != PE_MODEL_NAME or value['label'] not in (0, 1):
        raise ValueError('Invalid model or label')
    if any(value[k] != record[k] for k in ('video_id', 'label', 'group_id')):
        raise ValueError('Identity does not match manifest')
    for key, shape in (('features', (SEQUENCE_LENGTH, feature_dim)), ('gend_logits', (SEQUENCE_LENGTH, 2))):
        tensor = value[key]
        if tuple(tensor.shape) != shape or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f'Invalid {key}: shape {tuple(tensor.shape)}')
    ids = torch.as_tensor(value['frame_ids'])
    if ids.shape != (SEQUENCE_LENGTH,) or ids[0] < 0 or not torch.all(ids[1:] > ids[:-1]):
        raise ValueError(f'Frame IDs must be {SEQUENCE_LENGTH} strictly increasing non-negative integers')


def load_clip_frame_ids(record):
    """The 32 frame IDs the CLIP GRU-32 reads: every other one of the 64 stored per video."""
    path = master_path(FEATURE_ROOT, record)
    if not path.is_file():
        return None
    return torch.load(path, map_location='cpu', weights_only=True)['frame_ids'].tolist()[::2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', help='Process all videos')
    parser.add_argument('--groups', type=int, default=2, help='Smoke test: first N TRAIN groups')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.groups <= 204 or args.batch_size < 1:
        parser.error('groups must be 1..204 and batch-size must be positive')
    manifest, split = load_definitions()
    selected = set(split['groups']['train'][:args.groups])
    records = [r for r in manifest['videos'] if args.all or r['group_id'] in selected]
    from detector import prepare_model
    from src.hf.modeling_gend import GenD
    from temporal.pe_frames import load_aligned_frames_at_ids
    from temporal.video_loader import load_aligned_frame_sequence
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device, flush=True)
    face_detector = prepare_model(det_thres=0.1)
    model, info = GenD.from_pretrained(PE_MODEL_NAME, output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError(f'Incomplete pretrained load: {info}')
    model = model.to(device).eval().requires_grad_(False)
    feature_dim = model.feature_extractor.get_features_dim()
    print('PE feature dimension:', feature_dim, flush=True)

    summary = dict(successful=0, failed=0, skipped_valid=0, regenerated_invalid=0)
    failures, detect_s, encode_s = [], 0.0, 0.0
    started = time.perf_counter()
    for i, record in enumerate(records, 1):
        print(f'[{i}/{len(records)}] processing {record["video_id"]}.mp4', flush=True)
        path = master_path(PE_FEATURE_ROOT, record)
        if path.exists():
            try:
                validate_pe(torch.load(path, map_location='cpu', weights_only=True), record, feature_dim)
                summary['skipped_valid'] += 1
                continue
            except Exception:
                summary['regenerated_invalid'] += 1
        frame_ids = []
        try:
            t0 = time.perf_counter()
            clip_ids = load_clip_frame_ids(record)
            if clip_ids is not None:
                # Same moments as CLIP GRU-32: read those exact frames in one pass.
                frames, frame_ids = load_aligned_frames_at_ids(
                    str(ROOT / record['path']), face_detector, clip_ids, scale=1.3, target_size=(224, 224))
            else:
                # No CLIP file (real/029): search as CLIP did (64 frames), keep every other one.
                frames, frame_ids = load_aligned_frame_sequence(
                    str(ROOT / record['path']), face_detector, sequence_length=2 * SEQUENCE_LENGTH,
                    scale=1.3, target_size=(224, 224))
                frames, frame_ids = frames[::2], frame_ids[::2]
            t1 = time.perf_counter()
            features, logits = encode_frames(model, frames, device, args.batch_size)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            detect_s, encode_s = detect_s + (t1 - t0), encode_s + (t2 - t1)
            if not check_frame_alignment(record['video_id'], list(frame_ids), clip_ids):
                raise ValueError('PE and CLIP frame IDs differ')
            value = dict(video_id=record['video_id'], label=record['label'],
                         group_id=record['group_id'], frame_ids=torch.tensor(frame_ids),
                         features=features, gend_logits=logits, model_name=PE_MODEL_NAME)
            validate_pe(value, record, feature_dim)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.pt.tmp')
            torch.save(value, temporary)
            validate_pe(torch.load(temporary, map_location='cpu', weights_only=True), record, feature_dim)
            temporary.replace(path)
            summary['successful'] += 1
        except Exception as error:
            match = re.search(r'Only found (\d+) usable', str(error))
            failures.append(dict(video_id=record['video_id'], group_id=record['group_id'],
                                 reason=str(error), usable_frames=int(match[1]) if match else len(frame_ids)))
            summary['failed'] += 1
            print('FAILED:', error, flush=True)
    done = summary['successful'] + summary['failed']
    total = time.perf_counter() - started
    timing = dict(videos_timed=done, total_seconds=round(total, 1),
                  frame_loading_seconds=round(detect_s, 1), pe_encoding_seconds=round(encode_s, 1),
                  seconds_per_video=round(total / done, 2) if done else None)
    report = dict(scope='all' if args.all else 'smoke', model=PE_MODEL_NAME, feature_dim=feature_dim,
                  summary=summary, timing=timing, failures=failures)
    save_json(RESULT_ROOT / ('pe_preprocessing_all.json' if args.all else 'pe_preprocessing_smoke.json'), report)
    print(summary, timing)
    if failures:
        raise SystemExit('Preprocessing failures recorded; do not train on an incomplete dataset')


if __name__ == '__main__':
    main()
