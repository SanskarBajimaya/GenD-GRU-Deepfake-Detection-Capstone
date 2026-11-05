from torch.utils.data import Subset


# --------------------------------------------------
# Paired development groups
# --------------------------------------------------

PAIRED_GROUPS = {
    "group_1": [
        "183",
        "253",
        "183_253",
        "253_183",
    ],

    "group_2": [
        "469",
        "481",
        "469_481",
        "481_469",
    ],

    "group_3": [
        "585",
        "599",
        "585_599",
        "599_585",
    ],

    "group_4": [
        "672",
        "720",
        "672_720",
        "720_672",
    ],

    "group_5": [
        "866",
        "878",
        "866_878",
        "878_866",
    ],
}


# --------------------------------------------------
# Split definition
# --------------------------------------------------

TRAIN_GROUPS = [
    "group_1",
    "group_2",
    "group_3",
]

VAL_GROUPS = [
    "group_4",
]

TEST_GROUPS = [
    "group_5",
]


def _get_video_names(group_names):
    """
    Convert group names into a set of video names.
    """

    names = set()

    for group_name in group_names:
        names.update(
            PAIRED_GROUPS[group_name]
        )

    return names


def _get_indices(dataset, video_names):
    """
    Find dataset indices belonging to specified videos.
    """

    indices = []

    for index, sample in enumerate(dataset.samples):

        if sample["video"] in video_names:
            indices.append(index)

    return indices


def create_paired_splits(dataset):
    """
    Create train / validation / test splits while keeping
    related real and fake videos together.

    Returns:
        train_dataset
        val_dataset
        test_dataset
    """

    train_names = _get_video_names(TRAIN_GROUPS)
    val_names = _get_video_names(VAL_GROUPS)
    test_names = _get_video_names(TEST_GROUPS)

    # --------------------------------------------------
    # Safety checks
    # --------------------------------------------------

    if train_names & val_names:
        raise RuntimeError(
            "Train and validation videos overlap."
        )

    if train_names & test_names:
        raise RuntimeError(
            "Train and test videos overlap."
        )

    if val_names & test_names:
        raise RuntimeError(
            "Validation and test videos overlap."
        )

    all_expected = (
        train_names
        | val_names
        | test_names
    )

    dataset_names = {
        sample["video"]
        for sample in dataset.samples
    }

    missing = all_expected - dataset_names

    if missing:
        raise RuntimeError(
            f"Missing expected videos: {sorted(missing)}"
        )

    unexpected = dataset_names - all_expected

    if unexpected:
        raise RuntimeError(
            f"Unexpected videos found: {sorted(unexpected)}"
        )

    # --------------------------------------------------
    # Build indices
    # --------------------------------------------------

    train_indices = _get_indices(
        dataset,
        train_names,
    )

    val_indices = _get_indices(
        dataset,
        val_names,
    )

    test_indices = _get_indices(
        dataset,
        test_names,
    )

    # --------------------------------------------------
    # Create subsets
    # --------------------------------------------------

    train_dataset = Subset(
        dataset,
        train_indices,
    )

    val_dataset = Subset(
        dataset,
        val_indices,
    )

    test_dataset = Subset(
        dataset,
        test_indices,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )