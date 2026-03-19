"""Parse docking output files (Vina, Glide, AutoDock)."""

def parse_vina_log(path: str) -> dict:
    """Parse AutoDock Vina log file."""
    scores = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4 and parts[0].isdigit():
                try:
                    scores.append(float(parts[1]))
                except ValueError:
                    pass
    return {"best_score": min(scores) if scores else 0.0, "all_scores": scores}

def parse_vina_pdbqt(path: str) -> list:
    """Extract poses and energies from Vina output PDBQT."""
    models = []
    current = {"energy": None, "lines": []}
    with open(path) as f:
        for line in f:
            if line.startswith("MODEL"):
                current = {"energy": None, "lines": []}
            elif line.startswith("REMARK VINA RESULT"):
                parts = line.split()
                current["energy"] = float(parts[3])
            elif line.startswith("ENDMDL"):
                models.append(current)
    return models
