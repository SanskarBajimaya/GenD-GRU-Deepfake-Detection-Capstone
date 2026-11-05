"""Validated master artifacts; both models share the same frame indices."""
from pathlib import Path
import torch
from torch.utils.data import Dataset
from temporal.experiment_config import FEATURE_ROOT, MODEL_NAME


def validate_master(value, record=None):
    if not isinstance(value, dict):
        raise ValueError('Master must be a dictionary')
    for field in ('video_id', 'label', 'group_id', 'frame_ids', 'features', 'gend_logits', 'model_name'):
        if field not in value:
            raise ValueError(f'Missing {field}')
    if value['model_name'] != MODEL_NAME or value['label'] not in (0, 1):
        raise ValueError('Invalid model or label')
    if record and any(value[k] != record[k] for k in ('video_id', 'label', 'group_id')):
        raise ValueError('Master identity does not match manifest')
    for key, shape in (('features', (64, 1024)), ('gend_logits', (64, 2))):
        tensor = value[key]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f'Invalid {key} shape')
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f'Invalid {key} dtype or nonfinite values')
    ids = torch.as_tensor(value['frame_ids'])
    if ids.shape != (64,) or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError('Expected 64 integer frame IDs')
    if ids[0] < 0 or not torch.all(ids[1:] > ids[:-1]):
        raise ValueError('Frame IDs must be strictly chronological and unique')
    return value


def master_path(root, record):
    return root / ('fake' if record['label'] else 'real') / (record['video_id'] + '.pt')


def load_master(path, record=None):
    return validate_master(torch.load(path, map_location='cpu', weights_only=True), record)


class TemporalMasterFeatureDataset(Dataset):
    def __init__(self, records, sequence_length=64, root=FEATURE_ROOT):
        if sequence_length not in (32, 64):
            raise ValueError('Only 32 and 64 frames are supported')
        self.records, self.root = records, Path(root)
        self.indices = torch.arange(0, 64, 2 if sequence_length == 32 else 1)
        for record in records:
            if not master_path(self.root, record).is_file():
                raise ValueError(f'Missing master: {record["video_id"]}; complete preprocessing first')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        value = load_master(master_path(self.root, record), record)
        return dict(features=value['features'][self.indices].float(),
                    gend_logits=value['gend_logits'][self.indices].float(),
                    frame_ids=torch.as_tensor(value['frame_ids'])[self.indices],
                    label=torch.tensor(record['label'], dtype=torch.float32),
                    video_id=record['video_id'], group_id=record['group_id'])
