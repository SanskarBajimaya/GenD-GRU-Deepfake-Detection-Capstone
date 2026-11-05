"""Strict pairing verification and write-once group split (no ML dependencies)."""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from temporal.experiment_config import DATA_ROOT, ROOT, MANIFEST, SPLIT, SPLIT_SEED


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def scan(data_root):
    real_dir = data_root / 'original_sequences/youtube/c23/videos'
    fake_dir = data_root / 'manipulated_sequences/Deepfakes/c23/videos'
    real = {p.stem: p for p in sorted(real_dir.glob('*.mp4'))}
    fake = {p.stem: p for p in sorted(fake_dir.glob('*.mp4'))}
    if any(not re.fullmatch(r'\d{3}', name) for name in real):
        raise ValueError('Malformed original filename')
    pairs = set()
    for name in fake:
        if not re.fullmatch(r'\d{3}_\d{3}', name):
            raise ValueError(f'Malformed fake filename: {name}')
        a, b = name.split('_')
        if a == b:
            raise ValueError(f'Self pair: {name}')
        pairs.add(tuple(sorted((a, b))))
    groups, videos, used = [], [], set()
    for a, b in sorted(pairs):
        group_id = f'{a}_{b}'
        if used.intersection((a, b)):
            raise ValueError(f'Original belongs to multiple groups: {group_id}')
        used.update((a, b))
        names = [a, b, f'{a}_{b}', f'{b}_{a}']
        for label, name in zip((0, 0, 1, 1), names):
            paths = fake if label else real
            if name not in paths:
                raise ValueError(f'Missing original/reciprocal fake: {name}')
            path = paths[name].resolve()
            try:
                stored_path = path.relative_to(ROOT).as_posix()
            except ValueError:
                stored_path = path.as_posix()
            videos.append(dict(group_id=group_id, video_id=name, label=label,
                               class_name='FAKE' if label else 'REAL', path=stored_path,
                               source_ids=[a, b], group_members=names))
        groups.append(dict(group_id=group_id, real=[a, b], fake=names[2:]))
    if used != set(real) or (len(groups), len(real), len(fake)) != (255, 510, 510):
        raise ValueError(f'Expected 255 groups / 510 real / 510 fake; got '
                         f'{len(groups)} / {len(real)} / {len(fake)}; unused={set(real)-used}')
    return dict(schema_version=1, groups=groups, videos=videos)


def validate_split(manifest, split):
    if split['split_seed'] != SPLIT_SEED or split['manifest_hash'] != fingerprint(manifest):
        raise ValueError('Split seed or manifest fingerprint mismatch')
    sets = []
    for name, count in zip(('train', 'validation', 'test'), (204, 26, 25)):
        ids = split['groups'][name]
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError(f'Invalid {name} group count')
        sets.append(set(ids))
    if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('Groups overlap')
    if set.union(*sets) != {g['group_id'] for g in manifest['groups']}:
        raise ValueError('Split does not cover manifest')


def load_definitions():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    split = json.loads(SPLIT.read_text(encoding='utf-8'))
    validate_split(manifest, split)
    return manifest, split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    manifest = scan(args.data_root)
    if MANIFEST.exists():
        if json.loads(MANIFEST.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Existing manifest differs; refusing to replace it')
    else:
        save_json(MANIFEST, manifest)
    if SPLIT.exists():
        split = json.loads(SPLIT.read_text(encoding='utf-8'))
    else:
        ids = [g['group_id'] for g in manifest['groups']]
        random.Random(SPLIT_SEED).shuffle(ids)
        split = dict(split_seed=SPLIT_SEED, manifest_hash=fingerprint(manifest),
                     groups=dict(train=ids[:204], validation=ids[204:230], test=ids[230:]))
        validate_split(manifest, split)
        save_json(SPLIT, split)
    validate_split(manifest, split)
    print('Verified: 255 groups, 510 REAL, 510 FAKE, 1020 videos')
    for name, ids in split['groups'].items():
        print(f'{name}: {len(ids)} groups, {4*len(ids)} videos, {2*len(ids)} per class')
    print('No group/source/video overlap; saved split reused unchanged.')


if __name__ == '__main__':
    main()
