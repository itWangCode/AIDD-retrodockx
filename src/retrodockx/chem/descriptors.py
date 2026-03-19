"""Molecular descriptor computation."""
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import numpy as np

def compute_descriptors(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        "mw":           round(Descriptors.MolWt(mol), 3),
        "logp":         round(Descriptors.MolLogP(mol), 3),
        "tpsa":         round(Descriptors.TPSA(mol), 3),
        "hbd":          rdMolDescriptors.CalcNumHBD(mol),
        "hba":          rdMolDescriptors.CalcNumHBA(mol),
        "rings":        rdMolDescriptors.CalcNumRings(mol),
        "rot_bonds":    rdMolDescriptors.CalcNumRotatableBonds(mol),
        "arom_rings":   rdMolDescriptors.CalcNumAromaticRings(mol),
        "frac_csp3":    round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        "stereocenters":rdMolDescriptors.CalcNumStereocenters(mol),
        "lipinski_ok": (Descriptors.MolWt(mol) <= 500 and
                        Descriptors.MolLogP(mol) <= 5 and
                        rdMolDescriptors.CalcNumHBD(mol) <= 5 and
                        rdMolDescriptors.CalcNumHBA(mol) <= 10),
    }
