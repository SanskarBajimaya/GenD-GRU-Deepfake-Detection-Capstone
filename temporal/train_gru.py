import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from temporal.feature_dataset import TemporalFeatureDataset
from temporal.dataset_split import create_paired_splits
from temporal.model_a import TemporalGRU


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 42

BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-3

HIDDEN_DIM = 128

MODEL_OUTPUT = "temporal/models/best_gru.pt"


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch in loader:

            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            logits = model(features)

            loss = criterion(
                logits,
                labels,
            )

            total_loss += loss.item()

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= 0.5
            ).float()

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = (
        total_loss / len(loader)
    )

    accuracy = (
        correct / total
    )

    return average_loss, accuracy


# --------------------------------------------------
# Training
# --------------------------------------------------

def main():

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

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

    print()
    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))

    # --------------------------------------------------
    # DataLoaders
    # --------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # IMPORTANT:
    # We do NOT use test_dataset during training.

    # --------------------------------------------------
    # GRU
    # --------------------------------------------------

    model = TemporalGRU(
        feature_dim=1024,
        hidden_dim=HIDDEN_DIM,
    )

    model = model.to(device)

    print()
    print(model)

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    criterion = nn.BCEWithLogitsLoss()

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------
    # Track best validation model
    # --------------------------------------------------

    best_val_loss = float("inf")

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        model.train()

        total_train_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:

            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(features)

            # Calculate loss
            loss = criterion(
                logits,
                labels,
            )

            # Backpropagation
            loss.backward()

            # Update GRU parameters
            optimizer.step()

            total_train_loss += loss.item()

            # ------------------------------------------
            # Training accuracy
            # ------------------------------------------

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= 0.5
            ).float()

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = (
            total_train_loss
            / len(train_loader)
        )

        train_accuracy = (
            correct / total
        )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.3f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.3f}"
        )

        # --------------------------------------------------
        # Save best model
        # --------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            from pathlib import Path

            Path(
                MODEL_OUTPUT
            ).parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "feature_dim": 1024,

                    "hidden_dim":
                        HIDDEN_DIM,

                    "sequence_length": 16,

                    "best_val_loss":
                        best_val_loss,
                },
                MODEL_OUTPUT,
            )

            print(
                f"  -> Saved best model "
                f"(Val Loss: {val_loss:.4f})"
            )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("Best validation loss:", best_val_loss)
    print("Best model saved to:", MODEL_OUTPUT)


if __name__ == "__main__":
    main()