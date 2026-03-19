"""Parse MD trajectory analysis outputs (GROMACS, AMBER, NAMD)."""
import numpy as np

def parse_rmsd_xvg(path: str) -> dict:
    """Parse GROMACS RMSD .xvg file."""
    times, rmsds = [], []
    with open(path) as f:
        for line in f:
            if line.startswith(('#', '@')):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    times.append(float(parts[0]))
                    rmsds.append(float(parts[1]))
                except ValueError:
                    pass
    arr = np.array(rmsds) if rmsds else np.array([0.0])
    return {
        "time_ns":   times,
        "rmsd_nm":   rmsds,
        "mean_rmsd": float(arr.mean()),
        "max_rmsd":  float(arr.max()),
        "std_rmsd":  float(arr.std()),
        "stability": float(1.0 / (1.0 + arr.mean())),
    }

def parse_rmsf_xvg(path: str) -> dict:
    """Parse GROMACS RMSF .xvg file (per-residue flexibility)."""
    residues, rmsfs = [], []
    with open(path) as f:
        for line in f:
            if line.startswith(('#', '@')):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    residues.append(int(parts[0]))
                    rmsfs.append(float(parts[1]))
                except ValueError:
                    pass
    arr = np.array(rmsfs) if rmsfs else np.array([0.0])
    return {"residues": residues, "rmsf_nm": rmsfs,
            "mean_rmsf": float(arr.mean()), "max_rmsf": float(arr.max())}
