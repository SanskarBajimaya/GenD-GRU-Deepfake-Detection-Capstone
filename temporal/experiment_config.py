"""Paths and controlled parameters for the paired 32/64-frame experiment."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / 'data/faceforensic'
MANIFEST = ROOT / 'temporal/manifests/video_manifest.json'
SPLIT = ROOT / 'temporal/manifests/split_seed_2026.json'
FEATURE_ROOT = ROOT / 'temporal/features_64'
MODEL_ROOT = ROOT / 'temporal/models'
RESULT_ROOT = ROOT / 'temporal/results'
MODEL_NAME = 'yermandy/GenD_CLIP_L_14'
SPLIT_SEED = 2026
TRAIN_SEEDS = tuple(range(5))
MASTER_SEQUENCE_LENGTH = 64
FEATURE_DIM = 1024
