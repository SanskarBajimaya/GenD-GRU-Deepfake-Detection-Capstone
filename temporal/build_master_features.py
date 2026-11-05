"""Frozen GenD preprocessing. Default is a two TRAIN-group smoke test."""
import argparse
import re
import torch
from temporal.build_manifest import load_definitions, save_json
from temporal.experiment_config import ROOT, FEATURE_ROOT, MODEL_NAME, RESULT_ROOT
from temporal.master_dataset import load_master, master_path, validate_master


@torch.inference_mode()
def encode_frames(model, frames, device, batch_size):
    features, logits = [], []
    for start in range(0, len(frames), batch_size):
        batch = torch.stack([model.feature_extractor.preprocess(f)
                             for f in frames[start:start+batch_size]]).to(device)
        encoded = model.feature_extractor(batch)
        original_logits = model.model.forward(encoded)
        features.append(encoded.float().cpu())
        logits.append(original_logits.float().cpu())
    return torch.cat(features), torch.cat(logits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', help='Explicitly process all 1020 videos')
    parser.add_argument('--groups', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.groups <= 204 or args.batch_size < 1:
        parser.error('groups must be 1..204 and batch-size must be positive')
    manifest, split = load_definitions()
    selected = set(split['groups']['train'][:args.groups])
    records = [r for r in manifest['videos'] if args.all or r['group_id'] in selected]
    # Import expensive dependencies only after CLI and definitions are checked.
    from detector import prepare_model
    from src.hf.modeling_gend import GenD
    from temporal.video_loader import load_aligned_frame_sequence
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device, flush=True)
    face_detector = prepare_model(det_thres=0.1)
    model, info = GenD.from_pretrained(MODEL_NAME, output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError(f'Incomplete pretrained load: {info}')
    model = model.to(device).eval().requires_grad_(False)
    summary = dict(successful=0, failed=0, skipped_valid=0, regenerated_invalid=0)
    failures = []
    for i, record in enumerate(records, 1):
        print(f'[{i}/{len(records)}] processing {record["video_id"]}.mp4', flush=True)
        path = master_path(FEATURE_ROOT, record)
        if path.exists():
            try:
                load_master(path, record)
                summary['skipped_valid'] += 1
                continue
            except Exception:
                summary['regenerated_invalid'] += 1
        frame_ids = []
        try:
            frames, frame_ids = load_aligned_frame_sequence(
                str(ROOT / record['path']), face_detector, sequence_length=64,
                scale=1.3, target_size=(224, 224))
            features, logits = encode_frames(model, frames, device, args.batch_size)
            value = dict(video_id=record['video_id'], label=record['label'],
                         group_id=record['group_id'], frame_ids=torch.tensor(frame_ids),
                         features=features, gend_logits=logits, model_name=MODEL_NAME)
            validate_master(value, record)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.pt.tmp')
            torch.save(value, temporary)
            load_master(temporary, record)
            temporary.replace(path)
            summary['successful'] += 1
        except Exception as error:
            match = re.search(r'Only found (\d+) usable', str(error))
            failures.append(dict(video_id=record['video_id'], group_id=record['group_id'],
                                 reason=str(error), usable_frames=int(match[1]) if match else len(frame_ids)))
            summary['failed'] += 1
            print('FAILED:', error, flush=True)
    report = dict(scope='all' if args.all else 'smoke', summary=summary, failures=failures)
    save_json(RESULT_ROOT / ('preprocessing_all.json' if args.all else 'preprocessing_smoke.json'), report)
    print(summary)
    if failures:
        raise SystemExit('Preprocessing failures recorded; do not train on an incomplete dataset')


if __name__ == '__main__':
    main()
