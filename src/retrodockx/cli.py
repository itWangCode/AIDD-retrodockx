"""src/retrodockx/cli.py — unified CLI entry point."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

def main():
    import argparse
    parser = argparse.ArgumentParser(
        prog="retrodockx",
        description="RetroDock-X: ML-powered retrosynthesis + docking-guided route reranking",
    )
    sub = parser.add_subparsers(dest="cmd", help="Sub-commands")

    # run
    p_run = sub.add_parser("run", help="Run full pipeline")
    p_run.add_argument("--smiles",  type=str)
    p_run.add_argument("--name",    default="Target")
    p_run.add_argument("--batch",   type=str)
    p_run.add_argument("--pdb",     type=str)
    p_run.add_argument("--backend", default="rdkit", choices=["rdkit","askcos","aizynth"])
    p_run.add_argument("--no_shap", action="store_true")
    p_run.add_argument("--output",  default="analysis")

    # download
    p_dl = sub.add_parser("download", help="Download PDB files")
    p_dl.add_argument("--pdb",  type=str)
    p_dl.add_argument("--pdbs", type=str)
    p_dl.add_argument("--list", type=str)
    p_dl.add_argument("--out",  default="data/raw")

    args = parser.parse_args()

    if args.cmd == "run":
        os.system(f"python scripts/run_single.py "
                  f"{'--smiles ' + repr(args.smiles) if args.smiles else ''} "
                  f"{'--name '  + args.name if args.name else ''} "
                  f"{'--batch ' + args.batch if args.batch else ''} "
                  f"{'--pdb '   + args.pdb   if args.pdb   else ''} "
                  f"--backend {args.backend} "
                  f"{'--no_shap' if args.no_shap else ''} "
                  f"--output {args.output}")
    elif args.cmd == "download":
        os.system(f"python scripts/download_pdb.py "
                  f"{'--pdb '  + args.pdb  if args.pdb  else ''} "
                  f"{'--pdbs ' + args.pdbs if args.pdbs else ''} "
                  f"{'--list ' + args.list if args.list else ''} "
                  f"--out {args.out}")
    else:
        parser.print_help()
