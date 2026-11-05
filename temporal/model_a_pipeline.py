import torch
import torch.nn as nn

from temporal.model_a import TemporalGRU


class GenDGRU(nn.Module):

    def __init__(
        self,
        gend_model,
        hidden_dim=128
    ):
        super().__init__()

        self.gend = gend_model

        # Freeze GenD
        for parameter in self.gend.parameters():
            parameter.requires_grad = False

        feature_dim = (
            self.gend
            .feature_extractor
            .get_features_dim()
        )

        self.temporal_model = TemporalGRU(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim
        )

    def forward(self, frames):

        """
        frames:
        [batch, sequence, channels, height, width]
        """

        batch_size = frames.size(0)
        sequence_length = frames.size(1)

        # Convert:
        #
        # [B, T, C, H, W]
        #
        # into:
        #
        # [B*T, C, H, W]

        flat_frames = frames.reshape(
            batch_size * sequence_length,
            *frames.shape[2:]
        )

        # GenD is frozen
        with torch.no_grad():

            features = (
                self.gend
                .feature_extractor(flat_frames)
            )

        # Convert features back into sequences:
        #
        # [B*T, feature_dim]
        #
        # becomes
        #
        # [B, T, feature_dim]

        features = features.reshape(
            batch_size,
            sequence_length,
            -1
        )

        # Send sequence to GRU
        logits = self.temporal_model(features)

        return logits