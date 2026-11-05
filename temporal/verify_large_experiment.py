"""Fast contract tests, plus optional validation of saved master artifacts."""
import argparse
import copy
import tempfile
from pathlib import Path
import torch
from temporal.build_manifest import load_definitions, validate_split
from temporal.build_master_features import encode_frames
from temporal.experiment_config import MODEL_NAME, FEATURE_ROOT
from temporal.master_dataset import validate_master, load_master, master_path, TemporalMasterFeatureDataset
from temporal.large_experiment import metrics, checkpoint_path, evaluate
from torch.utils.data import DataLoader
from temporal.model_a import TemporalGRU


def must_fail(action):
    try:
        action()
    except (ValueError, KeyError):
        return
    raise AssertionError('Invalid input was accepted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='Require all eight smoke artifacts')
    parser.add_argument('--all', action='store_true', help='Validate all 1020 saved masters')
    args = parser.parse_args()
    manifest, split = load_definitions()
    validate_split(manifest, split)
    bad = copy.deepcopy(split)
    bad['groups']['test'][0] = bad['groups']['train'][0]
    must_fail(lambda: validate_split(manifest, bad))
    record = manifest['videos'][0]
    value = dict(video_id=record['video_id'], label=record['label'], group_id=record['group_id'],
                 model_name=MODEL_NAME, features=torch.randn(64, 1024),
                 gend_logits=torch.randn(64, 2), frame_ids=torch.arange(64)*3)
    validate_master(value, record)
    for key in ('features', 'gend_logits'):
        bad = copy.deepcopy(value)
        bad[key][0, 0] = float('nan')
        must_fail(lambda: validate_master(bad, record))
        bad[key][0, 0] = float('inf')
        must_fail(lambda: validate_master(bad, record))
    bad = copy.deepcopy(value)
    bad['frame_ids'][1] = bad['frame_ids'][0]
    must_fail(lambda: validate_master(bad, record))
    bad = copy.deepcopy(value)
    bad['group_id'] = 'wrong'
    must_fail(lambda: validate_master(bad, record))
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = master_path(root, record)
        path.parent.mkdir()
        torch.save(value, path)
        for length in (32, 64):
            data = TemporalMasterFeatureDataset([record], length, root)
            sample = data[0]
            indices = torch.arange(0, 64, 2 if length == 32 else 1)
            for key in ('features', 'gend_logits', 'frame_ids'):
                assert torch.equal(sample[key], value[key][indices])
            model = TemporalGRU(1024, 128)
            logit = model(sample['features'].unsqueeze(0))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, sample['label'].reshape(1))
            loss.backward()
            assert model.gru.weight_ih_l0.grad is not None
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def preprocess(self, frame):
            return frame
        def forward(self, inputs):
            self.calls += 1
            return inputs
    class FakeGenD(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_extractor = Encoder()
            self.model = torch.nn.Linear(1024, 2)
    fake = FakeGenD().eval()
    frames = [torch.randn(1024) for _ in range(64)]
    features, logits = encode_frames(fake, frames, 'cpu', 16)
    assert fake.feature_extractor.calls == 4
    with torch.inference_mode():
        assert torch.allclose(logits, fake.model(features), atol=1e-6)
    assert len({checkpoint_path(length, seed) for length in (32, 64) for seed in range(5)}) == 10
    assert metrics([0, 1], [0.1, 0.9])['auroc'] == 1
    assert metrics([0, 1], [0.5, 0.5])['auroc'] == 0.5
    must_fail(lambda: metrics([0, 0], [0.1, 0.2]))
    print('Contract checks PASS: split, corruption rejection, shared indices, backward pass, single encoder pass, metrics, ten checkpoint names')
    if args.smoke or args.all:
        groups = set(split['groups']['train'][:2])
        records = [r for r in manifest['videos'] if args.all or r['group_id'] in groups]
        for record in records:
            load_master(master_path(FEATURE_ROOT, record), record)
        print(f'Actual artifact checks PASS: {len(records)} videos')
        if args.smoke:
            first, second = split['groups']['train'][:2]
            training = TemporalMasterFeatureDataset([r for r in records if r['group_id'] == first], 32)
            validation = TemporalMasterFeatureDataset([r for r in records if r['group_id'] == second], 32)
            model = TemporalGRU(1024, 128)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            batch = next(iter(DataLoader(training, batch_size=4)))
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(batch['features']), batch['label'])
            loss.backward()
            optimizer.step()
            for length in (32, 64):
                validation = TemporalMasterFeatureDataset(validation.records, length)
                data = DataLoader(validation, batch_size=4)
                evaluate(data, 'cpu', model)
                evaluate(data, 'cpu')
            print('Smoke train/validation compatibility PASS (two TRAIN groups only; no experiment checkpoint saved)')


if __name__ == '__main__':
    main()
