import torch


@torch.no_grad()
def extract_gend_features(gend_model, frame_batch):
    """
    Extract features from GenD without using
    GenD's original classification head.

    frame_batch:
        [number_of_frames, channels, height, width]

    returns:
        [number_of_frames, feature_dim]
    """

    gend_model.eval()

    features = gend_model.feature_extractor(
        frame_batch
    )

    return features