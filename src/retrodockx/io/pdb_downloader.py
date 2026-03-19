"""src/retrodockx/io/pdb_downloader.py"""
import os, requests, time
from typing import List

def download_pdb(pdb_id: str, save_dir: str = "data/raw") -> str:
    os.makedirs(save_dir, exist_ok=True)
    pdb_id = pdb_id.upper().strip()
    out = os.path.join(save_dir, f"{pdb_id}.pdb")
    if os.path.exists(out):
        print(f"  [CACHE] {pdb_id}.pdb")
        return out
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    r = requests.get(url, timeout=30)
    if r.status_code == 200:
        with open(out, "w") as f:
            f.write(r.text)
        print(f"  [OK] Downloaded {pdb_id}.pdb")
        return out
    raise ValueError(f"PDB {pdb_id} not found (HTTP {r.status_code})")

def batch_download_pdb(pdb_ids: List[str], save_dir: str = "data/raw",
                        delay: float = 0.5) -> dict:
    results = {}
    for i, pid in enumerate(pdb_ids, 1):
        print(f"  [{i}/{len(pdb_ids)}] {pid}")
        try:
            results[pid] = {"status": "ok", "path": download_pdb(pid, save_dir)}
        except Exception as e:
            results[pid] = {"status": "error", "error": str(e)}
        time.sleep(delay)
    return results

def extract_ligand_smiles(pdb_path: str) -> List[str]:
    """Extract non-solvent HETATM residue names from PDB."""
    skip = {"HOH","WAT","NA","CL","MG","ZN","CA","K","SO4","PO4","GOL"}
    residues = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("HETATM"):
                res = line[17:20].strip()
                if res not in skip:
                    residues.add(res)
    return list(residues)
