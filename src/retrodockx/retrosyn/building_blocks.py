"""
src/retrodockx/retrosyn/building_blocks.py
==========================================
Curated purchasable building blocks and helper functions.

Design goals
------------
1. Keep purchasable logic separate from route generation logic.
2. Canonicalize all SMILES once to avoid mismatch issues.
3. Allow optional loading from a plain text / CSV-like file later.
4. Provide a practical heuristic fallback for very small molecules.

This is a starter in-repo catalog, not a replacement for a real
commercial building-block database (eMolecules / Enamine / Mcule / etc.).
"""

from __future__ import annotations

import os
from typing import Iterable, Optional, Set, List

from rdkit import Chem


def canon_smiles(smiles: str) -> Optional[str]:
    """Canonicalize SMILES with RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def safe_mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Safely parse a SMILES."""
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


# A practical starter set:
# - common aromatics
# - common alcohols / phenols
# - common amines
# - common acids / acyl donors
# - common medicinal-chemistry fragments
PURCHASABLE_RAW: Set[str] = {
    # ─────────────────────────────
    # Very common small molecules
    # ─────────────────────────────
    "CO",                 # methanol
    "CCO",                # ethanol
    "CCCO",               # propanol
    "CC(C)O",             # isopropanol
    "O",                  # water
    "N",                  # ammonia
    "CC(=O)O",            # acetic acid
    "CCC(=O)O",           # propionic acid
    "CC(C)(C)O",          # tBuOH
    "CC(C)=O",            # acetone
    "CC=O",               # acetaldehyde
    "O=CC1=CC=CC=C1",     # alt benzaldehyde style not canonical, will be canonized
    "O=Cc1ccccc1",        # benzaldehyde
    "O=CO",               # formic acid style alt
    "O=CC",               # acetaldehyde alt

    # ─────────────────────────────
    # Simple aromatics / heteroaromatics
    # ─────────────────────────────
    "c1ccccc1",           # benzene
    "Cc1ccccc1",          # toluene
    "Clc1ccccc1",         # chlorobenzene
    "Brc1ccccc1",         # bromobenzene
    "Ic1ccccc1",          # iodobenzene
    "Fc1ccccc1",          # fluorobenzene
    "c1ccncc1",           # pyridine
    "c1nccnc1",           # pyrimidine
    "c1cncnc1",           # pyrazine-like alt
    "c1ccoc1",            # furan
    "c1ccsc1",            # thiophene
    "c1cc[nH]c1",         # pyrrole

    # ─────────────────────────────
    # Common phenols / alcohols / ethers
    # ─────────────────────────────
    "Oc1ccccc1",          # phenol
    "Oc1ccc(O)cc1",       # hydroquinone / resorcinol family representative
    "OCc1ccccc1",         # benzyl alcohol
    "CCOc1ccccc1",        # phenetole-like precursor
    "COc1ccccc1",         # anisole
    "CCOC(=O)c1ccccc1",   # ethyl benzoate
    "COC(=O)c1ccccc1",    # methyl benzoate

    # ─────────────────────────────
    # Amines / anilines / amino alcohols
    # ─────────────────────────────
    "Nc1ccccc1",          # aniline
    "c1ccc(N)cc1",        # aniline alt
    "NCc1ccccc1",         # benzylamine
    "NCCO",               # ethanolamine
    "NCC(=O)O",           # glycine
    "CCN",                # ethylamine
    "CCCN",               # propylamine
    "CC(C)N",             # isopropylamine
    "CN",                 # methylamine
    "NCCc1ccccc1",        # phenethylamine
    "Nc1ccc(O)cc1",       # aminophenol isomer representative
    "Nc1ccc(Cl)cc1",      # chloroaniline representative
    "Nc1ccc(C)cc1",       # toluidine representative

    # ─────────────────────────────
    # Carboxylic acids / acyl donors / activated derivatives
    # ─────────────────────────────
    "O=C(O)c1ccccc1",     # benzoic acid
    "O=C(O)c1ccc(O)cc1",  # hydroxybenzoic acid / salicylic acid family
    "O=C(O)c1ccc(N)cc1",  # aminobenzoic acid
    "CC(=O)Cl",           # acetyl chloride
    "O=C(Cl)c1ccccc1",    # benzoyl chloride
    "CC(=O)OC(C)=O",      # acetic anhydride
    "O=C(OCC)c1ccccc1",   # ethyl benzoate alt
    "CCOC(=O)C",          # ethyl acetate
    "CC(=O)OCc1ccccc1",   # benzyl acetate
    "CC(=O)OCC",          # ethyl acetate alt
    "O=C(O)CCl",          # chloroacetic acid
    "O=C(Cl)CCl",         # chloroacetyl chloride

    # ─────────────────────────────
    # Halides / alkylating agents
    # ─────────────────────────────
    "BrC",                # bromomethane
    "BrCC",               # bromoethane
    "BrCCC",              # bromopropane
    "BrCc1ccccc1",        # benzyl bromide
    "ClCc1ccccc1",        # benzyl chloride
    "ICc1ccccc1",         # benzyl iodide
    "BrCCO",              # 2-bromoethanol
    "ClCCO",              # 2-chloroethanol

    # ─────────────────────────────
    # Boronic acids / coupling fragments
    # ─────────────────────────────
    "OB(O)c1ccccc1",      # phenylboronic acid alt
    "B(O)(O)c1ccccc1",    # phenylboronic acid
    "B(O)(O)c1ccc(F)cc1",
    "B(O)(O)c1ccc(Cl)cc1",
    "B(O)(O)c1ccc(C)cc1",
    "B(O)(O)c1ccncc1",

    # ─────────────────────────────
    # Sulfonylation / heterocycle-adjacent fragments
    # ─────────────────────────────
    "NS(=O)(=O)c1ccccc1",     # benzenesulfonamide
    "O=S(=O)(Cl)c1ccccc1",    # benzenesulfonyl chloride
    "O=S(=O)(Cl)c1ccc(Cl)cc1",
    "O=S(=O)(Cl)c1ccc(F)cc1",
    "Cc1onc(N)cc1",           # isoxazole-like representative
    "c1ncc([N+](=O)[O-])cc1", # nitro azaarene representative

    # ─────────────────────────────
    # Common benchmark / medchem-like compounds
    # ─────────────────────────────
    "CC(=O)Nc1ccccc1",        # acetanilide
    "CCOC(=O)c1ccc(N)cc1",    # benzocaine
    "CC(=O)NCc1ccccc1",       # N-benzylacetamide
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CC(=O)Nc1ccc(O)cc1",     # paracetamol
}


def canonicalize_smiles_set(smiles_iter: Iterable[str]) -> Set[str]:
    """Canonicalize a collection of SMILES, dropping invalid entries."""
    out: Set[str] = set()
    for s in smiles_iter:
        cs = canon_smiles(s)
        if cs is not None:
            out.add(cs)
    return out


PURCHASABLE: Set[str] = canonicalize_smiles_set(PURCHASABLE_RAW)


def load_building_blocks_from_file(path: str) -> Set[str]:
    """
    Load extra building blocks from a text/CSV-like file.

    Supported simple formats:
    - one SMILES per line
    - CSV with SMILES in first column
    - blank lines / comment lines starting with # are ignored
    """
    extra: Set[str] = set()
    if not path or not os.path.exists(path):
        return extra

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            token = raw.split(",")[0].strip()
            cs = canon_smiles(token)
            if cs is not None:
                extra.add(cs)
    return extra


def get_purchasable_set(extra_file: Optional[str] = None) -> Set[str]:
    """
    Return the curated purchasable set, optionally merged with an external file.
    """
    if extra_file:
        return set(PURCHASABLE) | load_building_blocks_from_file(extra_file)
    return set(PURCHASABLE)


def is_small_obviously_purchasable(smiles: str) -> bool:
    """
    Heuristic fallback for very small molecules.
    """
    mol = safe_mol_from_smiles(smiles)
    if mol is None:
        return False

    heavy = mol.GetNumHeavyAtoms()
    if heavy <= 3:
        return True

    return False


def is_purchasable(smiles: str, purchasable_set: Optional[Set[str]] = None) -> bool:
    """
    Check purchasability with:
    1) canonicalization
    2) small-molecule heuristic
    3) membership in curated / provided set
    """
    cs = canon_smiles(smiles)
    if cs is None:
        return False

    if is_small_obviously_purchasable(cs):
        return True

    blocks = purchasable_set if purchasable_set is not None else PURCHASABLE
    return cs in blocks


def get_default_building_blocks() -> List[str]:
    """Return sorted canonical purchasable entries for inspection/debugging."""
    return sorted(PURCHASABLE)
