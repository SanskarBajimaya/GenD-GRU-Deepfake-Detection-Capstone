"""Integrity checks and descriptive reporting for the fixed in-domain TEST split."""
import hashlib
import json
from collections import Counter

from temporal.large_experiment import *
from temporal.build_manifest import save_json


def verify_test_integrity(manifest, split, checkpoints, decision):
    if decision['sequence_length'] != 32:
        raise ValueError('This final experiment requires the pre-selected GRU-32')
    if (decision['manifest_hash'], decision['split_hash']) != (fingerprint(manifest), fingerprint(split)):
        raise ValueError('Saved selection fingerprints differ')
    groups = set(split['groups']['test'])
    records = [r for r in manifest['videos'] if r['group_id'] in groups]
    if len(groups) != 25 or len(records) != 100 or Counter(r['label'] for r in records) != {0: 50, 1: 50}:
        raise ValueError('Expected 25 TEST groups, 100 videos, 50 per class')
    if groups.intersection(split['groups']['train']) or groups.intersection(split['groups']['validation']):
        raise ValueError('TEST groups overlap development groups')
    if len({r['video_id'] for r in records}) != 100 or any(v != 4 for v in Counter(r['group_id'] for r in records).values()):
        raise ValueError('Invalid TEST video identities or group membership')
    for (length, seed), (cp, model) in checkpoints.items():
        digest = hashlib.sha256(checkpoint_path(length, seed).read_bytes()).hexdigest()
        if decision['checkpoint_hashes'][f'{length}_{seed}'] != digest:
            raise ValueError('Checkpoint changed since validation selection')
        if not all(torch.isfinite(t).all() for t in model.state_dict().values()):
            raise ValueError('Nonfinite checkpoint weights')
        if not 1 <= cp['best_epoch'] <= 20 or not all(np.isfinite(cp[k]) for k in ('best_val_auroc', 'best_val_bce', 'best_val_accuracy')):
            raise ValueError('Invalid best-epoch metadata')
    for length in (32, 64):
        dataset = TemporalMasterFeatureDataset(records, length)
        for index, record in enumerate(records):
            master = load_master(master_path(FEATURE_ROOT, record), record)
            sample = dataset[index]
            stride = 2 if length == 32 else 1
            for key in ('features', 'gend_logits'):
                if not torch.equal(sample[key], master[key][::stride].float()):
                    raise ValueError('Frame consistency check failed')
            if not torch.equal(sample['frame_ids'], torch.as_tensor(master['frame_ids'])[::stride]):
                raise ValueError('Frame IDs differ from master')
    result = dict(test_groups=25, test_videos=100, real=50, fake=50,
                  disjoint_groups=True, master_artifacts_valid=True,
                  frame_consistency_verified=True, all_ten_checkpoints_valid=True,
                  checkpoint_hashes=decision['checkpoint_hashes'])
    print('TEST integrity PASS: 25 disjoint groups, 100 videos (50 REAL / 50 FAKE), valid masters, shared frames, ten unchanged checkpoints')
    return result


def save_test_report(rows, predictions, summary, integrity, decision):
    tests = [r for r in rows if r['split'] == 'test']
    if len(tests) != 12 or len(predictions) != 1200:
        raise ValueError('Expected twelve TEST evaluations and 1200 per-video predictions')
    error_analysis = []
    for length in (32, 64):
        baseline = {r['video_id'] for r in predictions if r['model'] == 'GenD' and r['sequence_length'] == length and not r['correct']}
        seeds = []
        for seed in TRAIN_SEEDS:
            subset = [r for r in predictions if r['model'] == 'GRU' and r['sequence_length'] == length and r['seed'] == seed]
            if len(subset) != 100 or len({r['video_id'] for r in subset}) != 100:
                raise ValueError('Each seed must contain exactly 100 unique videos')
            values = metrics([r['true_label'] for r in subset], [r['fake_probability'] for r in subset])
            saved = next(r for r in tests if r['model'] == 'GRU' and r['sequence_length'] == length and r['training_seed'] == seed)
            if any(values[k] != saved[k] for k in values):
                raise ValueError('Prediction metrics differ from reported metrics')
            errors = {r['video_id'] for r in subset if not r['correct']}
            seeds.append(dict(seed=seed, gru_errors=sorted(errors), shared_errors=sorted(errors & baseline),
                              fixed_by_gru=sorted(baseline - errors), new_gru_errors=sorted(errors - baseline)))
        counts = Counter(video for seed in seeds for video in seed['gru_errors'])
        error_analysis.append(dict(sequence_length=length, gend_errors=sorted(baseline), seeds=seeds,
                                   gru_error_seed_counts=dict(sorted(counts.items())),
                                   interpretation='Per-seed errors; no ensemble or best TEST seed selection'))
    deltas = []
    for length in (32, 64):
        baseline = next(r for r in summary if r['sequence_length'] == length and r['model'] == 'GenD')
        gru = next(r for r in summary if r['sequence_length'] == length and r['model'] == 'GRU')
        deltas.append(dict(sequence_length=length, **{k: gru[k + '_mean'] - baseline[k + '_mean'] for k in METRICS}))
    final = dict(selected_model='GRU-32', selection_basis='validation_mean_AUROC', split_seed=SPLIT_SEED,
                 training_seeds=list(TRAIN_SEEDS), test_groups=25, test_videos=100, threshold=0.5,
                 standard_deviation_ddof=1, integrity=integrity, test_summary=summary, differences=deltas,
                 error_analysis=error_analysis, checkpoints_unchanged=True, retraining=False,
                 model_reselection=False, test_based_tuning=False, ensembling=False,
                 statistical_significance_test_performed=False, cross_dataset_generalization_tested=False,
                 scope='In-domain FaceForensics++ c23; pretrained GenD was trained on FaceForensics++ c23 (experiment context supplied by user)')
    write_csv(RESULT_ROOT / 'test_predictions.csv', predictions)
    write_csv(RESULT_ROOT / 'test_metrics.csv', tests)
    save_json(RESULT_ROOT / 'test_integrity.json', integrity)
    save_json(RESULT_ROOT / 'test_error_analysis.json', error_analysis)
    save_json(RESULT_ROOT / 'final_experiment_summary.json', final)
    lines = ['# Final in-domain TEST report', '',
             'Exactly 100 previously unevaluated TEST videos (25 groups, 50 REAL and 50 FAKE) were evaluated once per model/run. All five seeds are reported independently; no ensemble was created.', '',
             '## PRIMARY FINAL RESULT', '',
             'GRU-32 was selected before TEST exclusively by mean validation AUROC. This decision is unchanged.', '']
    def table(selected):
        output = ['| Model | Frames | AUROC | Accuracy | BCE | Precision | Recall | F1 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in selected:
            cells = [(f"{row[k+'_mean']:.6f} +/- {row[k+'_std']:.6f}" if row['model'] == 'GRU'
                      else f"{row[k+'_mean']:.6f}") for k in METRICS]
            output.append(f"| {row['model']} | {row['sequence_length']} | " + ' | '.join(cells) + ' |')
        return output
    lines += table([r for r in summary if r['sequence_length'] == 32])
    for delta in deltas:
        if delta['sequence_length'] == 64:
            lines += ['', '## Secondary 64-frame result', ''] + table([r for r in summary if r['sequence_length'] == 64])
        lines += ['', f"GRU-{delta['sequence_length']} minus GenD-{delta['sequence_length']}: AUROC {delta['auroc']:+.6f}; accuracy {delta['accuracy']:+.6f}; BCE {delta['bce']:+.6f}. Lower BCE is better."]
    lines += ['', 'GRU-32 remains the pre-selected model. The 64-frame experiment is a secondary comparison. Standard deviations are sample standard deviations across five independent seeds (ddof=1); accuracy is a fraction.', '',
              '## Individual results and confusion counts', '',
              '| Model | Frames | Seed | AUROC | Accuracy | BCE | Precision | Recall | F1 | TP | TN | FP | FN |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in tests:
        cells = [f'{row[k]:.6f}' for k in METRICS] + [str(row[k]) for k in ('tp', 'tn', 'fp', 'fn')]
        lines.append(f"| {row['model']} | {row['sequence_length']} | {row['training_seed']} | " + ' | '.join(cells) + ' |')
    lines += ['', '## Descriptive error analysis', '', 'No aggregate GRU prediction is defined. Shared, fixed, and new errors are therefore reported separately for each seed.', '']
    def ids(values):
        return ', '.join(values) if values else 'none'
    for analysis in error_analysis:
        lines += [f"### {analysis['sequence_length']} frames", '', f"GenD errors: {ids(analysis['gend_errors'])}.", '',
                  '| GRU seed | GRU errors | Shared errors | GenD errors fixed | New GRU errors |',
                  '|---:|---|---|---|---|']
        for seed in analysis['seeds']:
            lines.append(f"| {seed['seed']} | {ids(seed['gru_errors'])} | {ids(seed['shared_errors'])} | {ids(seed['fixed_by_gru'])} | {ids(seed['new_gru_errors'])} |")
        lines += ['', 'GRU error counts across five seeds: ' + (', '.join(f'{v}: {n}/5' for v, n in analysis['gru_error_seed_counts'].items()) or 'none') + '.', '']
    primary = deltas[0]
    lines += ['## Interpretation and limitations', '',
              f"For the pre-selected GRU-32, mean accuracy changes by {100*primary['accuracy']:+.2f} percentage points (equivalent to {100*primary['accuracy']:+.2f} correctly classified videos per 100, averaged across seeds). AUROC changes by {primary['auroc']:+.6f} and BCE by {primary['bce']:+.6f}. These are descriptive numerical differences; no statistical significance test was performed.", '',
              'The five-seed means and sample standard deviations describe training-seed variability, not uncertainty across datasets or source groups. The per-seed error table shows whether individual improvements are consistent.', '',
              'This is an in-domain FaceForensics++ c23 test. As specified in the experiment context, pretrained GenD was itself trained on FaceForensics++ c23; this result does not establish cross-dataset generalization. External evaluation has not been performed.', '',
              'All ten checkpoint hashes and the saved validation decision were verified before TEST inference. No retraining, checkpoint modification, epoch reselection, model reselection, threshold tuning, hyperparameter tuning, preprocessing, feature regeneration, or ensembling occurred. Stored logits supply the frozen GenD baseline.', '',
              'Structured outputs: test_predictions.csv, test_metrics.csv, test_summary.csv, experiment_results.csv (preserves validation rows), test_integrity.json, test_error_analysis.json, final_experiment_summary.json. The existing selected_configuration.json and checkpoints are preserved.']
    (RESULT_ROOT / 'test_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('Saved TEST predictions, metrics, descriptive errors and final report; pre-selected GRU-32 unchanged.')
