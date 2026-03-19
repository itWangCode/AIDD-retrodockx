"""Extract docking features for ML models."""
import numpy as np

def extract_docking_features(docking_result: dict) -> np.ndarray:
    """Convert docking result dict to feature vector."""
    score = docking_result.get("best_score", 0.0)
    poses = docking_result.get("all_scores", [score])
    return np.array([
        score,
        abs(score),
        min(poses) if poses else score,
        np.mean(poses) if poses else score,
        np.std(poses) if len(poses) > 1 else 0.0,
        len(poses),
    ])
