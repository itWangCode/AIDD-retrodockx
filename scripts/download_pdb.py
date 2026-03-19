"""
scripts/download_pdb.py
========================
Download PDB files (single or batch).

Usage:
  python scripts/download_pdb.py --pdb 1ABC
  python scripts/download_pdb.py --pdbs 1ABC,2HHB,4HHB
  python scripts/download_pdb.py --list data/input/pdb_ids.txt
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
from retrodockx.io.pdb_downloader import download_pdb, batch_download_pdb


def main():
    parser = argparse.ArgumentParser(description="PDB Downloader")
    parser.add_argument("--pdb",  type=str, help="Single PDB ID")
    parser.add_argument("--pdbs", type=str, help="Comma-separated PDB IDs")
    parser.add_argument("--list", type=str, help="Text file with one PDB ID per line")
    parser.add_argument("--out",  type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()

    ids = []
    if args.pdb:
        ids = [args.pdb]
    elif args.pdbs:
        ids = [p.strip() for p in args.pdbs.split(",") if p.strip()]
    elif args.list:
        with open(args.list) as f:
            ids = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    else:
        parser.print_help()
        return

    if len(ids) == 1:
        path = download_pdb(ids[0], save_dir=args.out)
        print(f"Saved: {path}")
    else:
        results = batch_download_pdb(ids, save_dir=args.out)
        ok = sum(1 for v in results.values() if v["status"] == "ok")
        print(f"\nSummary: {ok}/{len(ids)} downloaded to {args.out}")

if __name__ == "__main__":
    main()
