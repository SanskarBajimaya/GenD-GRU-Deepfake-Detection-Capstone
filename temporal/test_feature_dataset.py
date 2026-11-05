from torch.utils.data import DataLoader

from temporal.feature_dataset import TemporalFeatureDataset


def main():

    # -----------------------------------------------
    # Create dataset
    # -----------------------------------------------

    dataset = TemporalFeatureDataset(
        feature_root="temporal/features",
        expected_sequence_length=16,
        expected_feature_dim=1024,
    )

    print()
    print("Dataset size:", len(dataset))

    # -----------------------------------------------
    # Look at one sample
    # -----------------------------------------------

    sample = dataset[0]

    print()
    print("First sample:")
    print("Video:", sample["video"])
    print("Label:", sample["label"])
    print(
        "Feature shape:",
        sample["features"].shape
    )

    # -----------------------------------------------
    # Create DataLoader
    # -----------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
    )

    # -----------------------------------------------
    # Test one batch
    # -----------------------------------------------

    batch = next(iter(loader))

    print()
    print("Batch videos:")
    print(batch["video"])

    print()
    print(
        "Batch feature shape:",
        batch["features"].shape
    )

    print(
        "Batch label shape:",
        batch["label"].shape
    )

    print(
        "Batch labels:",
        batch["label"]
    )

    print()
    print(
        "SUCCESS: feature dataset and DataLoader work."
    )


if __name__ == "__main__":
    main()