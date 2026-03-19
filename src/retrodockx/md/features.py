"""Extract MD features for ML models."""
import numpy as np

def extract_md_features(md_result: dict) -> np.ndarray:
    """Convert MD analysis dict to feature vector."""
    return np.array([
        md_result.get("mean_rmsd", 0.0),
        md_result.get("max_rmsd", 0.0),
        md_result.get("std_rmsd", 0.0),
        md_result.get("stability", 0.5),
    ])
