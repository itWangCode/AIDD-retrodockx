"""SMILES standardization via RDKit."""
from rdkit import Chem

def standardize(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else smiles

def is_valid(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None
