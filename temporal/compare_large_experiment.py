"""Validate all ten runs; persist selection before optional held-out testing."""
import argparse
import hashlib
import json
from temporal.large_experiment import *
from temporal.build_manifest import save_json


def summarize(rows):
    result = []
    for length in (32, 64):
        for model in ('GenD', 'GRU'):
            selected = [r for r in rows if r['sequence_length'] == length and r['model'] == model]
            row = dict(model=model, sequence_length=length, split=selected[0]['split'], runs=len(selected))
            for key in METRICS:
                values = [r[key] for r in selected]
                row[key + '_mean'] = float(np.mean(values))
                row[key + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            result.append(row)
            print(f'{model:4} {length}: AUROC {row["auroc_mean"]:.4f} +/- {row["auroc_std"]:.4f}; '
                  f'accuracy {row["accuracy_mean"]:.4f} +/- {row["accuracy_std"]:.4f}; '
                  f'BCE {row["bce_mean"]:.4f} +/- {row["bce_std"]:.4f}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', action='store_true', help='Evaluate TEST only after saved validation selection')
    args = parser.parse_args()
    decision_path = RESULT_ROOT / 'selected_configuration.json'
    if args.test and not decision_path.is_file():
        raise ValueError('Run validation comparison first to save configuration selection before TEST')
    manifest, split = load_definitions()
    preflight(manifest, split, ('validation', 'test') if args.test else ('validation',))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoints = {(length, seed): load_checkpoint(length, seed, manifest, split, device)
                   for length in (32, 64) for seed in TRAIN_SEEDS}
    predictions = []
    if args.test:
        from temporal.test_reporting import verify_test_integrity, save_test_report
        integrity = verify_test_integrity(manifest, split, checkpoints,
                                         json.loads(decision_path.read_text(encoding='utf-8')))
    rows = []
    for name in (('validation', 'test') if args.test else ('validation',)):
        if name == 'test':
            saved = json.loads(decision_path.read_text(encoding='utf-8'))
            if saved != decision:
                raise ValueError('Saved validation selection differs; test refused')
        split_rows = []
        for length in (32, 64):
            data = loader(manifest, split, name, length)
            split_rows.append(dict(model='GenD', sequence_length=length, training_seed='',
                                   split_seed=SPLIT_SEED, split=name, epoch='',
                                   **evaluate(data, device, prediction_rows=predictions if name == 'test' else None,
                                              context=dict(model='GenD', sequence_length=length, seed=''))))
            for seed in TRAIN_SEEDS:
                cp, model = checkpoints[length, seed]
                values = evaluate(data, device, model,
                                  prediction_rows=predictions if name == 'test' else None,
                                  context=dict(model='GRU', sequence_length=length, seed=seed))
                if name == 'validation' and not np.isclose(values['auroc'], cp['best_val_auroc'], atol=1e-8):
                    raise ValueError('Validation AUROC no longer reproduces checkpoint')
                split_rows.append(dict(model='GRU', sequence_length=length, training_seed=seed,
                                       split_seed=SPLIT_SEED, split=name, epoch=cp['best_epoch'], **values))
        print(name.upper())
        summary = summarize(split_rows)
        if name == 'validation':
            candidates = [r for r in summary if r['model'] == 'GRU']
            winner = max(candidates, key=lambda r: (r['auroc_mean'], -r['bce_mean'], r['accuracy_mean'], -r['sequence_length']))
            decision = dict(sequence_length=winner['sequence_length'],
                            rule='mean validation AUROC; lower BCE; higher accuracy; exact tie prefers 32',
                            manifest_hash=fingerprint(manifest), split_hash=fingerprint(split),
                            checkpoint_hashes={f'{length}_{seed}': hashlib.sha256(
                                checkpoint_path(length, seed).read_bytes()).hexdigest()
                                for length, seed in checkpoints})
            if not args.test:
                if decision_path.exists() and json.loads(decision_path.read_text(encoding='utf-8')) != decision:
                    raise ValueError('Existing validation decision belongs to different checkpoints; archive the results before a new experiment')
                save_json(decision_path, decision)
            print('Selected GRU frames:', winner['sequence_length'])
        for row in summary:
            row['selected_gru'] = row['model'] == 'GRU' and row['sequence_length'] == decision['sequence_length']
        if not (args.test and name == 'validation'):
            write_csv(RESULT_ROOT / f'{name}_summary.csv', summary)
        rows.extend(split_rows)
    # Preserve validation rows and all per-seed final metrics.
    result_path = RESULT_ROOT / 'experiment_results.csv'
    if args.test:
        with result_path.open(newline='', encoding='utf-8') as stream:
            saved_validation = [r for r in csv.DictReader(stream) if r['split'] == 'validation']
        if len(saved_validation) != 12:
            raise ValueError('Expected twelve saved validation rows')
        rows = saved_validation + [r for r in rows if r['split'] == 'test']
    if not args.test and result_path.exists():
        with result_path.open(newline='', encoding='utf-8') as stream:
            previous_test = [r for r in csv.DictReader(stream) if r['split'] == 'test']
        rows.extend(previous_test)
    write_csv(result_path, rows)
    if args.test:
        save_test_report(rows, predictions, summary, integrity, decision)
    print('Split groups:', {k: len(v) for k, v in split['groups'].items()})
    print('GenD results deterministic; GRU standard deviation uses ddof=1 across five seeds.')


if __name__ == '__main__':
    main()
