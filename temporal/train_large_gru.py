"""Run one specified controlled GRU training seed; never builds a test loader."""
import argparse
import time
from temporal.large_experiment import *


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence-length', type=int, choices=(32, 64), required=True)
    parser.add_argument('--seed', type=int, choices=TRAIN_SEEDS, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    path = checkpoint_path(args.sequence_length, args.seed)
    if path.exists():
        raise ValueError(f'{path} already exists; preserve or explicitly archive it before rerunning')
    manifest, split = load_definitions()
    preflight(manifest, split, ('train', 'validation'))
    seed_run(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = loader(manifest, split, 'train', args.sequence_length, args.seed, True)
    val = loader(manifest, split, 'validation', args.sequence_length)
    model = TemporalGRU(1024, 128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    best, rows = None, []
    for epoch in range(1, 21):
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
            rows.append(dict(model='GRU', sequence_length=args.sequence_length,
                             training_seed=args.seed, split_seed=SPLIT_SEED, split=name,
                             epoch=epoch, **values))
        score = (val_metrics['auroc'], -val_metrics['bce'])
        if best is None or score > best:
            best = score
            cp = dict(model_state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                      feature_dim=1024, hidden_dim=128, num_layers=1, dropout=0.0,
                      sequence_length=args.sequence_length, training_seed=args.seed,
                      split_seed=SPLIT_SEED, best_epoch=epoch, best_val_auroc=val_metrics['auroc'],
                      best_val_bce=val_metrics['bce'], best_val_accuracy=val_metrics['accuracy'],
                      manifest_hash=fingerprint(manifest), split_hash=fingerprint(split),
                      epochs=20, learning_rate=1e-3, batch_size=4, completed=False)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(cp, path)
        print(f'Epoch {epoch}/20 validation: {val_metrics}', flush=True)
        write_csv(RESULT_ROOT / f'gru{args.sequence_length}_seed{args.seed}_epochs.csv', rows)
    cp['completed'] = True
    cp['training_seconds'] = time.perf_counter() - started
    torch.save(cp, path)
    write_csv(RESULT_ROOT / f'gru{args.sequence_length}_seed{args.seed}_run.csv',
              [dict(model='GRU', sequence_length=args.sequence_length,
                    training_seed=args.seed, split_seed=SPLIT_SEED,
                    checkpoint_path=str(path), best_epoch=cp['best_epoch'],
                    best_val_auroc=cp['best_val_auroc'], best_val_bce=cp['best_val_bce'],
                    best_val_accuracy=cp['best_val_accuracy'],
                    training_seconds=cp['training_seconds'])])


if __name__ == '__main__':
    main()
