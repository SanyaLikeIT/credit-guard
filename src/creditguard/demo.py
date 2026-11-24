"""Synthetic demonstration profiles, never copied from dataset records."""

from creditguard.data import FEATURES

DEMO_PROFILES = {
    "lower": dict(zip(FEATURES, [0.08, 55, 0, 0.15, 6500, 8, 0, 1, 0, 0], strict=True)),
    "intermediate": dict(zip(FEATURES, [0.65, 40, 1, 0.45, 3500, 6, 0, 1, 0, 2], strict=True)),
    "higher": dict(zip(FEATURES, [1.1, 30, 3, 0.85, 1800, 4, 2, 0, 1, 2], strict=True)),
}
