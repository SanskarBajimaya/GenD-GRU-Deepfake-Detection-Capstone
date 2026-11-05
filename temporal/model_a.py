import torch
import torch.nn as nn


class TemporalGRU(nn.Module):
    """
    Model A:
    Takes a sequence of GenD feature vectors and
    produces one Real/Fake prediction for the video.
    """

    def __init__(
        self,
        feature_dim,
        hidden_dim=128,
        num_layers=1,
        dropout=0.0
    ):
        super().__init__()

        # Temporal model
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Video-level classifier
        self.classifier = nn.Linear(
            hidden_dim,
            1
        )

    def forward(self, features):
        """
        features shape:

        [batch_size, sequence_length, feature_dim]

        Example:
        [8, 16, 1024]

        8 videos
        16 frames per video
        1024 GenD features per frame
        """

        gru_output, hidden = self.gru(features)

        # hidden shape:
        # [num_layers, batch_size, hidden_dim]

        # Take final GRU hidden state
        video_feature = hidden[-1]

        # Convert temporal representation
        # into one Real/Fake logit
        logit = self.classifier(video_feature)

        return logit.squeeze(-1)