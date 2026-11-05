"""Zero-shot Celeb-DF v2 test of the CLIP GRU-32, PE GRU-32 and their probability average.

Uses the 32 evenly spaced face crops per video that detector.py already wrote for the 518 official
test videos (one folder per video). Both backbones encode the same crops, so the two GRUs see the same
moments. Nothing here trains or tunes; thresholds other than 0.5 are reported only for reference.

    python -m temporal.eval_celebdf_gru --frames-root <GenD>/datasets/CDFv2-test/CDFv2
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from temporal.build_manifest import load_definitions
from temporal.build_master_features import encode_frames
from temporal.build_pe_features import PE_MODEL_NAME
from temporal.experiment_config import MODEL_NAME as CLIP_MODEL_NAME, ROOT, TRAIN_SEEDS
from temporal.large_experiment import load_checkpoint
from temporal.model_a import TemporalGRU
from temporal.train_pe_gru import checkpoint_path as pe_checkpoint_path

OUT = ROOT / 'temporal/results_pe'
FRAMES = 32
LABELS = {'Celeb-real': 0, 'YouTube-real': 0, 'Celeb-synthesis': 1}


def list_videos(frames_root):
    videos = []
    for subset, label in LABELS.items():
        for folder in sorted((frames_root / subset).iterdir()):
            files = sorted(folder.glob('*.png'))
            if not 0 < len(files) <= FRAMES:
                raise ValueError(f'{folder} has {len(files)} crops, expected at most {FRAMES}')
            # 13 of 518 videos lost 1-2 crops to missed detections; repeat the last crop to keep all videos.
            files += [files[-1]] * (FRAMES - len(files))
            videos.append(dict(video_id=f'{subset}/{folder.name}', label=label, files=files))
    return videos


def extract(model_name, videos, device, cache):
    """(N, 32, 1024) features, cached so an interrupted run resumes."""
    if cache.exists():
        return torch.load(cache, weights_only=True)
    from src.hf.modeling_gend import GenD
    model = GenD.from_pretrained(model_name).to(device).eval().requires_grad_(False)
    out = []
    for i, v in enumerate(videos, 1):
        frames = [Image.open(f).convert('RGB') for f in v['files']]
        features, _ = encode_frames(model, frames, device, 16)
        out.append(features)
        if i % 50 == 0:
            print(f'{model_name}: {i}/{len(videos)}', flush=True)
    out = torch.stack(out)
    torch.save(out, cache)
    del model
    torch.cuda.empty_cache()
    return out


def seed_mean_probability(features, models, device):
    probs = []
    with torch.inference_mode():
        for model in models:
            model.eval()
            p = torch.cat([model(features[s:s + 32].to(device)).sigmoid().cpu() for s in range(0, len(features), 32)])
            probs.append(p.numpy())
    return np.mean(probs, axis=0)


def summarize(name, labels, scores):
    fpr, tpr, thr = roc_curve(labels, scores)
    eer_i = int(np.argmin(np.abs(fpr - (1 - tpr))))
    pred = scores >= 0.5
    return dict(model=name, auroc=roc_auc_score(labels, scores), ap=average_precision_score(labels, scores),
                eer=(fpr[eer_i] + 1 - tpr[eer_i]) / 2, recall_at_1pct_fpr=float(tpr[fpr <= 0.01].max()),
                recall_at_5pct_fpr=float(tpr[fpr <= 0.05].max()), accuracy_at_0p5=float((pred == labels).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--frames-root', type=Path, required=True)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    videos = list_videos(args.frames_root)
    labels = np.array([v['label'] for v in videos])
    print(f'{len(videos)} videos, {labels.sum()} fake', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    clip_x = extract(CLIP_MODEL_NAME, videos, device, OUT / 'celebdf_clip_features.pt')
    pe_x = extract(PE_MODEL_NAME, videos, device, OUT / 'celebdf_pe_features.pt')

    manifest, split = load_definitions()
    clip_models = [load_checkpoint(FRAMES, s, manifest, split, device)[1] for s in TRAIN_SEEDS]
    pe_models = []
    for s in TRAIN_SEEDS:
        cp = torch.load(pe_checkpoint_path(s), weights_only=True, map_location='cpu')
        m = TemporalGRU(1024, 128).to(device)
        m.load_state_dict(cp['model_state_dict'])
        pe_models.append(m)

    p_clip = seed_mean_probability(clip_x, clip_models, device)
    p_pe = seed_mean_probability(pe_x, pe_models, device)
    p_avg = (p_clip + p_pe) / 2
    rows = [summarize('CLIP GRU-32 (5 seeds)', labels, p_clip), summarize('PE GRU-32 (5 seeds)', labels, p_pe),
            summarize('CLIP-GRU + PE-GRU mean prob', labels, p_avg)]
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    with open(OUT / 'celebdf_gru_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    with open(OUT / 'celebdf_gru_predictions.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['video_id', 'true_label', 'clip_gru', 'pe_gru', 'mean'])
        for v, a, b, c in zip(videos, p_clip, p_pe, p_avg):
            w.writerow([v['video_id'], v['label'], a, b, c])


if __name__ == '__main__':
    main()
