"""Final comparison: released GenD heads, CLIP-GRU, PE-GRU and the ensembles on FF++ test and Celeb-DF v2 test.

Reads saved predictions only (nothing is trained or re-scored except the released PE per-frame head on the FF++
test features). Writes temporal/results_pe/final_comparison_{ffpp,celebdf}.csv and final_comparison_gains.csv.

    python -m temporal.compare_all --genD-root <GenD folder>
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from temporal.build_manifest import load_definitions
from temporal.build_pe_features import PE_FEATURE_ROOT
from temporal.experiment_config import ROOT
from temporal.master_dataset import master_path

OUT = ROOT / 'temporal/results_pe'
N_BOOT = 2000


def stats(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    i = int(np.argmin(np.abs(fpr - (1 - tpr))))
    return dict(auroc=roc_auc_score(y, s), ap=average_precision_score(y, s), eer=(fpr[i] + 1 - tpr[i]) / 2,
                recall_at_1pct_fpr=float(tpr[fpr <= 0.01].max()), recall_at_5pct_fpr=float(tpr[fpr <= 0.05].max()))


def bootstrap(y, scores, pairs):
    """Paired bootstrap over videos: same resample for every model. Returns CIs of AUROC and of AUROC differences."""
    rng = np.random.default_rng(0)
    n = len(y)
    auc = {k: [] for k in scores}
    diff = {p: [] for p in pairs}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        a = {k: roc_auc_score(y[idx], v[idx]) for k, v in scores.items()}
        for k in a:
            auc[k].append(a[k])
        for p in pairs:
            diff[p].append(a[p[0]] - a[p[1]])
    ci = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    return {k: ci(v) for k, v in auc.items()}, {p: ci(v) for p, v in diff.items()}


def ffpp_scores(genD_root):
    manifest, split = load_definitions()
    groups = set(split['groups']['test'])
    records = [r for r in manifest['videos'] if r['group_id'] in groups]
    ids = [r['video_id'] for r in records]
    y = np.array([r['label'] for r in records])
    pe_head, clip_head = [], []
    for r in records:
        v = torch.load(master_path(PE_FEATURE_ROOT, r), map_location='cpu', weights_only=True)
        pe_head.append(v['gend_logits'].softmax(-1)[..., 1].mean().item())
    tp = pd.read_csv(ROOT / 'temporal/results/test_predictions.csv', dtype={'video_id': str})
    tp = tp[tp.sequence_length == 32]
    gd = tp[tp.model == 'GenD'].set_index('video_id').fake_probability
    gru = tp[tp.model == 'GRU'].groupby('video_id').fake_probability.mean()
    pe_gru = pd.read_csv(OUT / 'gru32_pe_test_predictions.csv', dtype={'video_id': str}).set_index('video_id').fake_probability
    clip_head, clip_gru, pe_gru = (np.array([s[i] for i in ids]) for s in (gd, gru, pe_gru))
    pe_head = np.array(pe_head)
    return y, {'Released GenD-CLIP (per-frame)': clip_head, 'Released GenD-PE (per-frame)': pe_head,
               'Ensemble of released heads (CLIP+PE)': (clip_head + pe_head) / 2,
               'CLIP-GRU-32': clip_gru, 'PE-GRU-32': pe_gru, 'Ensemble of GRUs (CLIP-GRU+PE-GRU)': (clip_gru + pe_gru) / 2}


def celebdf_scores(genD_root):
    base = genD_root / 'results'
    ens = pd.read_csv(base / 'ensemble/CLIP+PE_mean_prob/video_predictions.csv').set_index('video')
    gr = pd.read_csv(OUT / 'celebdf_gru_predictions.csv').set_index('video_id').loc[ens.index]
    y = ens.label.to_numpy()
    return y, {'Released GenD-CLIP (per-frame)': ens.GenD_CLIP_L_14.to_numpy(),
               'Released GenD-PE (per-frame)': ens.GenD_PE_L.to_numpy(),
               'Ensemble of released heads (CLIP+PE)': ens.p_fake.to_numpy(),
               'CLIP-GRU-32': gr.clip_gru.to_numpy(), 'PE-GRU-32': gr.pe_gru.to_numpy(),
               'Ensemble of GRUs (CLIP-GRU+PE-GRU)': gr['mean'].to_numpy()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--genD-root', type=Path, required=True)
    args = parser.parse_args()
    gains = []
    for name, loader in (('ffpp', ffpp_scores), ('celebdf', celebdf_scores)):
        y, scores = loader(args.genD_root)
        keys = list(scores)
        pairs = [(keys[3], keys[0]), (keys[4], keys[1]), (keys[5], keys[2]), (keys[5], keys[3]), (keys[5], keys[4]),
                 (keys[2], keys[0]), (keys[2], keys[1])]
        auc_ci, diff_ci = bootstrap(y, scores, pairs)
        rows = [dict(model=k, videos=len(y), fakes=int(y.sum()), **stats(y, v), auroc_ci_low=auc_ci[k][0],
                     auroc_ci_high=auc_ci[k][1]) for k, v in scores.items()]
        pd.DataFrame(rows).to_csv(OUT / f'final_comparison_{name}.csv', index=False)
        for (a, b), (lo, hi) in diff_ci.items():
            gains.append(dict(dataset=name, a=a, b=b, auroc_diff=roc_auc_score(y, scores[a]) - roc_auc_score(y, scores[b]),
                              ci_low=lo, ci_high=hi))
        print(name, len(y), int(y.sum()))
        print(pd.DataFrame(rows).round(4).to_string())
    pd.DataFrame(gains).to_csv(OUT / 'final_comparison_gains.csv', index=False)
    print(pd.DataFrame(gains).round(4).to_string())


if __name__ == '__main__':
    main()
