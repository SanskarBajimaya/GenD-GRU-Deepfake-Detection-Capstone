from temporal.feature_dataset import TemporalFeatureDataset
from temporal.dataset_split import create_paired_splits


def print_split(name, subset):

    print()
    print("=" * 50)
    print(name)
    print("=" * 50)

    real_count = 0
    fake_count = 0

    for index in subset.indices:

        sample = subset.dataset.samples[index]

        video = sample["video"]
        label = sample["label"]

        label_name = (
            "REAL"
            if label == 0
            else "FAKE"
        )

        print(
            f"{video:<12} {label_name}"
        )

        if label == 0:
            real_count += 1
        else:
            fake_count += 1

    print()
    print("Total:", len(subset))
    print("Real:", real_count)
    print("Fake:", fake_count)


def main():

    dataset = TemporalFeatureDataset(
        feature_root="temporal/features",
        expected_sequence_length=16,
        expected_feature_dim=1024,
    )

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = create_paired_splits(dataset)

    print_split(
        "TRAIN",
        train_dataset,
    )

    print_split(
        "VALIDATION",
        val_dataset,
    )

    print_split(
        "TEST",
        test_dataset,
    )

    print()
    print("=" * 50)

    print(
        "Train size:",
        len(train_dataset),
    )

    print(
        "Validation size:",
        len(val_dataset),
    )

    print(
        "Test size:",
        len(test_dataset),
    )

    assert len(train_dataset) == 12
    assert len(val_dataset) == 4
    assert len(test_dataset) == 4

    print()
    print(
        "SUCCESS: paired dataset split is correct."
    )


if __name__ == "__main__":
    main()