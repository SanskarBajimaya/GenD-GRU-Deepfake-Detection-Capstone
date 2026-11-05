"""Train / report the PE GRU on 32 frames with the same protocol as the CLIP GRU-32.

Same split (seed 2026), model (1024 -> GRU(128) -> Linear(1)), optimizer (Adam 1e-3), loss, batch size (4),
20 epochs, seeds 0-4 and checkpoint rule (max validation AUROC, then lower BCE) as temporal/train_large_gru.py.
Only the input changes: PE features from temporal/features_pe32/ (built by temporal.build_pe_features --all).

    python -m temporal.train_pe_gru                          # train seeds 0-4 (train + validation only)
    python -m temporal.train_pe_gru --seeds 0 --epochs 2     # quick check; writes nothing permanent with --dry-run
    python -m temporal.train_pe_gru --report validation      # summary of the 5 saved checkpoints
    python -m temporal.train_pe_gru --report test            # run once, after validation, never to choose settings
"""
import argparse
import statistics
import time

import torch
from torch.utils.data import DataLoader, Dataset

from temporal.build_manifest import fingerprint, load_definitions
from temporal.build_pe_features import PE_FEATURE_ROOT, PE_MODEL_NAME, SEQUENCE_LENGTH, validate_pe
from temporal.experiment_config import FEATURE_DIM, ROOT, SPLIT_SEED, TRAIN_SEEDS
from temporal.large_experiment import metrics, seed_run, write_csv
from temporal.master_dataset import master_path
from temporal.model_a import TemporalGRU

PE_MODEL_ROOT = ROOT / 'temporal/models_pe'
PE_RESULT_ROOT = ROOT / 'temporal/results_pe'
EPOCHS, BATCH_SIZE, LEARNING_RATE, HIDDEN = 20, 4, 1e-3, 128


class PEFeatureDataset(Dataset):
    """All PE features of the chosen records, validated once and held in memory (about 130 KB per video)."""

    def __init__(self, records):
        self.items = []
        for record in records:
            path = master_path(PE_FEATURE_ROOT, record)
            if not path.is_file():
                raise ValueError(f'Missing PE features for {record["video_id"]}; run build_pe_features --all first')
            value = torch.load(path, map_location='cpu', weights_only=True)
            validate_pe(value, record, FEATURE_DIM)
            self.items.append(dict(features=value['features'].float(), gend_logits=value['gend_logits'].float(),
                                   label=torch.tensor(record['label'], dtype=torch.float32),
                                   video_id=record['video_id'], group_id=record['group_id']))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def make_loader(manifest, split, name, seed=0, shuffle=False):
    groups = set(split['groups'][name])
    records = [r for r in manifest['videos'] if r['group_id'] in groups]
    return DataLoader(PEFeatureDataset(records), batch_size=BATCH_SIZE, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)


def evaluate(loader, device, model=None, rows=None):
    """model=None scores with the released per-frame GenD-PE head, averaged over frames."""
    labels, probabilities = [], []
    if model is not None:
        model.eval()
    with torch.inference_mode():
        for batch in loader:
            p = (batch['gend_logits'].softmax(-1)[..., 1].mean(1) if model is None
                 else model(batch['features'].to(device)).sigmoid().cpu())
            labels.extend(batch['label'].tolist())
            probabilities.extend(p.tolist())
            if rows is not None:
                rows.extend(dict(video_id=v, group_id=g, true_label=int(y), fake_probability=q)
                            for v, g, y, q in zip(batch['video_id'], batch['group_id'], batch['label'].tolist(), p.tolist()))
    return metrics(labels, probabilities)


def checkpoint_path(seed):
    return PE_MODEL_ROOT / f'gru{SEQUENCE_LENGTH}_pe_seed{seed}_best.pt'


def train_seed(seed, manifest, split, device, epochs, dry_run):
    path = checkpoint_path(seed)
    if path.exists() and not dry_run:
        raise ValueError(f'{path} already exists; preserve or archive it before rerunning')
    started = time.perf_counter()
    seed_run(seed)
    train = make_loader(manifest, split, 'train', seed, True)
    val = make_loader(manifest, split, 'validation')
    model = TemporalGRU(FEATURE_DIM, HIDDEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()
    best, rows, checkpoint = None, [], None
    for epoch in range(1, epochs + 1):
        model.train()
        labels, probabilities = [], []
        for batch in train:
            optimizer.zero_grad()
            logits = model(batch['features'].to(device))
            loss = criterion(logits, batch['label'].to(device))
            loss.backward()
            optimizer.step()
            labels.extend(batch['label'].tolist())
            probabilities.extend(logits.detach().sigmoid().cpu().tolist())
        val_metrics = evaluate(val, device, model)
        for name, values in (('train', metrics(labels, probabilities)), ('validation', val_metrics)):
            rows.append(dict(model='GRU-PE', sequence_length=SEQUENCE_LENGTH, training_seed=seed,
                             split_seed=SPLIT_SEED, split=name, epoch=epoch, **values))
        score = (val_metrics['auroc'], -val_metrics['bce'])
        if best is None or score > best:
            best = score
            checkpoint = dict(model_state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                              feature_dim=FEATURE_DIM, hidden_dim=HIDDEN, num_layers=1, dropout=0.0,
                              sequence_length=SEQUENCE_LENGTH, training_seed=seed, split_seed=SPLIT_SEED,
                              best_epoch=epoch, best_val_auroc=val_metrics['auroc'], best_val_bce=val_metrics['bce'],
                              best_val_accuracy=val_metrics['accuracy'], feature_model=PE_MODEL_NAME,
                              manifest_hash=fingerprint(manifest), split_hash=fingerprint(split),
                              epochs=epochs, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, completed=False)
        print(f'seed {seed} epoch {epoch}/{epochs} validation AUROC {val_metrics["auroc"]:.5f} BCE {val_metrics["bce"]:.4f}', flush=True)
    if dry_run:
        print(f'dry run: nothing saved (best epoch {checkpoint["best_epoch"]})')
        return
    checkpoint['completed'] = True
    checkpoint['training_seconds'] = time.perf_counter() - started
    PE_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    write_csv(PE_RESULT_ROOT / f'gru{SEQUENCE_LENGTH}_pe_seed{seed}_epochs.csv', rows)


def report(split_name, manifest, split, device):
    loader = make_loader(manifest, split, split_name)
    results, per_video = [], {}
    for seed in TRAIN_SEEDS:
        cp = torch.load(checkpoint_path(seed), map_location='cpu', weights_only=True)
        if not cp.get('completed') or cp['manifest_hash'] != fingerprint(manifest) or cp['split_hash'] != fingerprint(split):
            raise ValueError(f'Checkpoint for seed {seed} is incomplete or does not match this split')
        model = TemporalGRU(FEATURE_DIM, HIDDEN).to(device)
        model.load_state_dict(cp['model_state_dict'])
        rows = []
        results.append(evaluate(loader, device, model, rows))
        for row in rows:
            per_video.setdefault(row['video_id'], dict(row, fake_probability=[]))['fake_probability'].append(row['fake_probability'])
    baseline = evaluate(loader, device, None)
    summary = []
    for name, runs in (('GRU-PE (mean of 5 seeds)', results), ('GenD-PE per-frame head', [baseline])):
        row = dict(model=name, split=split_name, sequence_length=SEQUENCE_LENGTH, runs=len(runs))
        for key in ('auroc', 'accuracy', 'bce', 'precision', 'recall', 'f1'):
            row[f'{key}_mean'] = statistics.mean(r[key] for r in runs)
            row[f'{key}_std'] = statistics.stdev(r[key] for r in runs) if len(runs) > 1 else 0.0
        summary.append(row)
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items() if k.endswith('_mean') or k == 'model'})
    write_csv(PE_RESULT_ROOT / f'gru{SEQUENCE_LENGTH}_pe_{split_name}_summary.csv', summary)
    # One probability per video (mean over seeds): the input for averaging with the CLIP GRU.
    write_csv(PE_RESULT_ROOT / f'gru{SEQUENCE_LENGTH}_pe_{split_name}_predictions.csv',
              [dict(video_id=v['video_id'], group_id=v['group_id'], true_label=v['true_label'],
                    fake_probability=statistics.mean(v['fake_probability'])) for v in per_video.values()])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seeds', type=int, nargs='+', choices=TRAIN_SEEDS, default=list(TRAIN_SEEDS))
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--dry-run', action='store_true', help='Train but save nothing')
    parser.add_argument('--report', choices=('validation', 'test'), help='Summarize the 5 saved checkpoints on a split')
    args = parser.parse_args()
    manifest, split = load_definitions()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.report:
        return report(args.report, manifest, split, device)
    for seed in args.seeds:
        train_seed(seed, manifest, split, device, args.epochs, args.dry_run)


if __name__ == '__main__':
    main()
