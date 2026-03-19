"""
scripts/run_batch.py
====================
Batch pipeline runner. Reads a CSV file and processes all molecules.

Usage:
  python scripts/run_batch.py --input data/input/molecules.csv
  python scripts/run_batch.py --input data/input/molecules.csv --backend aizynth --no_shap
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import pandas as pd
from run_single import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="RetroDock-X Batch Runner")
    parser.add_argument("--input",   required=True, help="Input CSV (columns: smiles, name)")
    parser.add_argument("--backend", default="rdkit", choices=["rdkit","askcos","aizynth"])
    parser.add_argument("--n_routes",type=int, default=5)
    parser.add_argument("--no_shap", action="store_true")
    parser.add_argument("--output",  default="batch")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df.columns = [c.lower().strip() for c in df.columns]

    smiles_list = df["smiles"].tolist()
    names = df["name"].tolist() if "name" in df.columns else [f"Mol_{i+1}" for i in range(len(df))]
    docking = df["docking_score"].tolist() if "docking_score" in df.columns else None

    run_pipeline(
        smiles_list, names,
        output_prefix=args.output,
        backend=args.backend,
        compare_models=True,
        run_shap=not args.no_shap,
        docking_scores=docking,
        n_routes=args.n_routes,
    )

if __name__ == "__main__":
    main()
