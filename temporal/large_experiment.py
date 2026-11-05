"""Shared reproducible training/evaluation utilities; no raw video inference."""
import csv
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, precision_score, recall_score, f1_score, confusion_matrix
from temporal.build_manifest import load_definitions, fingerprint
from temporal.experiment_config import *
from temporal.master_dataset import TemporalMasterFeatureDataset, load_master, master_path
from temporal.model_a import TemporalGRU

METRICS = ('auroc', 'accuracy', 'bce', 'precision', 'recall', 'f1')


def metrics(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=np.float64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError('Metrics require both classes')
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Invalid probabilities')
    predictions = probabilities >= 0.5
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return dict(auroc=float(roc_auc_score(labels, probabilities)),
                accuracy=float(accuracy_score(labels, predictions)),
                bce=float(log_loss(labels, probabilities, labels=[0, 1])),
                precision=float(precision_score(labels, predictions, zero_division=0)),
                recall=float(recall_score(labels, predictions, zero_division=0)),
                f1=float(f1_score(labels, predictions, zero_division=0)),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def seed_run(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def preflight(manifest, split=None, names=None):
    groups = (set().union(*(split['groups'][name] for name in names))
              if names is not None else None)
    for record in manifest['videos']:
        if groups is not None and record['group_id'] not in groups:
            continue
        load_master(master_path(FEATURE_ROOT, record), record)


def loader(manifest, split, name, length, seed=0, shuffle=False):
    groups = set(split['groups'][name])
    records = [r for r in manifest['videos'] if r['group_id'] in groups]
    return DataLoader(TemporalMasterFeatureDataset(records, length), batch_size=4,
                      shuffle=shuffle, generator=torch.Generator().manual_seed(seed), num_workers=0)


def evaluate(loader, device, model=None, prediction_rows=None, context=None):
    labels, probabilities = [], []
    if model is not None:
        model.eval()
    with torch.inference_mode():
        for batch in loader:
            p = (batch['gend_logits'].softmax(-1)[..., 1].mean(1) if model is None
                 else model(batch['features'].to(device)).sigmoid().cpu())
            labels.extend(batch['label'].tolist())
            probabilities.extend(p.tolist())
            if prediction_rows is not None:
                for i, probability in enumerate(p.tolist()):
                    label = int(batch['label'][i])
                    predicted = int(probability >= 0.5)
                    prediction_rows.append(dict(video_id=batch['video_id'][i],
                                                group_id=batch['group_id'][i], true_label=label,
                                                **context, fake_probability=probability,
                                                predicted_label=predicted, correct=predicted == label))
    return metrics(labels, probabilities)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('Empty result table')
    keys = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def checkpoint_path(length, seed):
    return MODEL_ROOT / f'gru{length}_seed{seed}_best.pt'


def load_checkpoint(length, seed, manifest, split, device):
    cp = torch.load(checkpoint_path(length, seed), weights_only=True, map_location='cpu')
    expected = dict(sequence_length=length, training_seed=seed, split_seed=SPLIT_SEED,
                    feature_dim=1024, hidden_dim=128, num_layers=1, dropout=0.0,
                    epochs=20, learning_rate=1e-3, batch_size=4,
                    manifest_hash=fingerprint(manifest), split_hash=fingerprint(split), completed=True)
    if any(cp.get(k) != v for k, v in expected.items()):
        raise ValueError(f'Incomplete or incompatible checkpoint for {length}/{seed}')
    model = TemporalGRU(1024, 128).to(device)
    model.load_state_dict(cp['model_state_dict'])
    return cp, model
