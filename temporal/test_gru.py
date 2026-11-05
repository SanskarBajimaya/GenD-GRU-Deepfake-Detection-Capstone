import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from temporal.feature_dataset import TemporalFeatureDataset
from temporal.dataset_split import create_paired_splits
from temporal.model_a import TemporalGRU


MODEL_PATH = "temporal/models/best_gru.pt"

BATCH_SIZE = 4


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # --------------------------------------------------
    # Load trained model
    # --------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    model = TemporalGRU(
        feature_dim=checkpoint["feature_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    # --------------------------------------------------
    # Test
    # --------------------------------------------------

    total_loss = 0.0
    correct = 0
    total = 0

    print()
    print("=" * 65)
    print("TEST RESULTS")
    print("=" * 65)

    with torch.no_grad():

        for batch in test_loader:

            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            logits = model(features)

            loss = criterion(
                logits,
                labels,
            )

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= 0.5
            ).float()

            total_loss += loss.item()

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            # ------------------------------------------
            # Print each video
            # ------------------------------------------

            for video, label, probability, prediction in zip(
                batch["video"],
                labels,
                probabilities,
                predictions,
            ):

                true_name = (
                    "FAKE"
                    if label.item() == 1
                    else "REAL"
                )

                predicted_name = (
                    "FAKE"
                    if prediction.item() == 1
                    else "REAL"
                )

                print(
                    f"{video:<12} "
                    f"True: {true_name:<5} | "
                    f"Predicted: {predicted_name:<5} | "
                    f"Fake probability: "
                    f"{probability.item():.4f}"
                )

    test_loss = total_loss / len(test_loader)

    test_accuracy = correct / total

    print()
    print("=" * 65)

    print(
        f"Test Loss: {test_loss:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy:.3f} "
        f"({correct}/{total})"
    )

    print("=" * 65)


if __name__ == "__main__":
    main()