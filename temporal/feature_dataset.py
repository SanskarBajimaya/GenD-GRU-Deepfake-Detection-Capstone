from pathlib import Path

import torch
from torch.utils.data import Dataset


class TemporalFeatureDataset(Dataset):
    """
    Loads precomputed GenD feature sequences.

    Expected structure:

    temporal/features/
        real/
            183.pt
            253.pt
            ...
        fake/
            183_253.pt
            253_183.pt
            ...

    Each .pt file should contain:
        [sequence_length, feature_dim]

    Example:
        [16, 1024]

    Labels:
        real = 0
        fake = 1
    """

    def __init__(
        self,
        feature_root="temporal/features",
        expected_sequence_length=16,
        expected_feature_dim=1024,
    ):
        self.feature_root = Path(feature_root)

        self.expected_sequence_length = expected_sequence_length
        self.expected_feature_dim = expected_feature_dim

        self.samples = []

        # -----------------------------------------------
        # REAL files
        # -----------------------------------------------

        real_dir = self.feature_root / "real"

        for path in sorted(real_dir.glob("*.pt")):
            self.samples.append(
                {
                    "path": path,
                    "label": 0.0,
                    "video": path.stem,
                }
            )

        # -----------------------------------------------
        # FAKE files
        # -----------------------------------------------

        fake_dir = self.feature_root / "fake"

        for path in sorted(fake_dir.glob("*.pt")):
            self.samples.append(
                {
                    "path": path,
                    "label": 1.0,
                    "video": path.stem,
                }
            )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No .pt feature files found in {self.feature_root}"
            )

        print(
            f"Loaded {len(self.samples)} feature sequences."
        )

        print(
            f"Real: {sum(s['label'] == 0 for s in self.samples)}"
        )

        print(
            f"Fake: {sum(s['label'] == 1 for s in self.samples)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        features = torch.load(
            sample["path"],
            map_location="cpu",
            weights_only=True,
        )

        # -----------------------------------------------
        # Validate tensor
        # -----------------------------------------------

        expected_shape = (
            self.expected_sequence_length,
            self.expected_feature_dim,
        )

        if tuple(features.shape) != expected_shape:
            raise ValueError(
                f"{sample['path']} has shape "
                f"{tuple(features.shape)}, "
                f"expected {expected_shape}"
            )

        if torch.isnan(features).any():
            raise ValueError(
                f"NaN found in {sample['path']}"
            )

        if torch.isinf(features).any():
            raise ValueError(
                f"Infinity found in {sample['path']}"
            )

        # Ensure correct datatype
        features = features.float()

        label = torch.tensor(
            sample["label"],
            dtype=torch.float32,
        )

        return {
            "features": features,
            "label": label,
            "video": sample["video"],
        }