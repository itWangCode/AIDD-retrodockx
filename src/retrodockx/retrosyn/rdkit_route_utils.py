"""
src/retrodockx/retrosyn/rdkit_route_utils.py
=============================================
RDKit-based route generation utilities used as fallback by both adapters.
Also used directly for: molecule standardization, route post-processing,
purchasability checks, SA score, and route graph construction.

This is NOT a standalone retrosynthesis engine — it's the fallback layer
and post-processing toolkit that wraps around the real backends.
"""

from __future__ import annotations

import math
import random
import hashlib
from typing import List, Dict, Optional, Tuple, Set

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from retrodockx.retrosyn.base import SynthesisRoute, ReactionStep


# ═══════════════════════════════════════════════════════
#  SMILES / CHEM HELPERS
# ═══════════════════════════════════════════════════════

def canon_smiles(smiles: str) -> Optional[str]:
    """Return canonical RDKit SMILES or None if parsing fails."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def safe_mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Parse a SMILES safely."""
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def dedupe_smiles(smiles_list: List[str]) -> List[str]:
    """Deduplicate by canonical SMILES while preserving order."""
    out: List[str] = []
    seen: Set[str] = set()
    for s in smiles_list:
        cs = canon_smiles(s)
        if cs is None or cs in seen:
            continue
        seen.add(cs)
        out.append(cs)
    return out


def is_reasonable_precursor(smiles: str) -> bool:
    """
    Very light precursor sanity filter.
    Avoid over-filtering: keep small legitimate building blocks like CO, CCO, etc.
    """
    mol = safe_mol_from_smiles(smiles)
    if mol is None:
        return False

    # Reject isolated single atoms / pathological fragments
    if mol.GetNumAtoms() <= 1:
        return False

    # Reject molecules with zero heavy atoms
    if mol.GetNumHeavyAtoms() == 0:
        return False

    return True


# ═══════════════════════════════════════════════════════
#  CURATED REACTION TEMPLATES
# ═══════════════════════════════════════════════════════

TEMPLATES = {
    # Ester -> acid + alcohol
    "ester_hydrolysis": {
        "smarts":     "[C:1](=[O:2])[O:3][C,c:4]>>[C:1](=[O:2])O.[O:3][C,c:4]",
        "class":      "Hydrolysis",
        "confidence": 0.90,
        "conditions": "H2O, H+ or OH-, heat",
    },

    # Amide -> acid + amine
    "amide_hydrolysis": {
        "smarts":     "[C:1](=[O:2])[N:3][C,c:4]>>[C:1](=[O:2])O.[N:3][C,c:4]",
        "class":      "Hydrolysis",
        "confidence": 0.85,
        "conditions": "H2O, H+ or OH-, heat",
    },

    # Simple N-acylation disconnection
    "n_acylation": {
        "smarts":     "[N:1][C:2](=[O:3])[C,c:4]>>[N:1].[C,c:4][C:2](=[O:3])Cl",
        "class":      "Acylation",
        "confidence": 0.88,
        "conditions": "Et3N, DCM, 0C to RT",
    },

    # ArOAc / ROAc -> alcohol or phenol
    "acetylation": {
        "smarts":     "[C,c:1][O:2][C:3](=[O:4])C>>[C,c:1][OH:2]",
        "class":      "Deprotection",
        "confidence": 0.88,
        "conditions": "Ac2O, pyridine or AcCl",
    },

    # Ether cleavage to alcohol + benzyl bromide style precursor
    "o_alkylation": {
        "smarts":     "[C,c:1][O:2][CH2:3][C,c:4]>>[C,c:1]O.[Br][CH2:3][C,c:4]",
        "class":      "Etherification",
        "confidence": 0.80,
        "conditions": "K2CO3, DMF, 60C",
    },

    # Carboxylic acid <- primary alcohol
    "oxidation_primary_alcohol": {
        "smarts":     "[C,c:1][C:2](=[O:3])O>>[C,c:1][CH2:2]O",
        "class":      "Oxidation",
        "confidence": 0.75,
        "conditions": "TEMPO/NaOCl or PCC",
    },

    # Secondary amine disconnection
    "reductive_amination": {
        "smarts":     "[C,c:1][NH:2][CH2:3][C,c:4]>>[C,c:1][NH2:2].[C,c:4][CH:3]=O",
        "class":      "Amination",
        "confidence": 0.78,
        "conditions": "NaBH3CN, MeOH",
    },

    # Biaryl disconnection
    "suzuki_biaryl": {
        "smarts":     "[c:1]-[c:2]>>[c:1]Br.[c:2]B(O)O",
        "class":      "C-C Coupling",
        "confidence": 0.85,
        "conditions": "Pd(PPh3)4, K2CO3, EtOH/H2O",
    },

    # Secondary ester disconnection
    "mitsunobu_ester": {
        "smarts":     "[C:1](=[O:2])[O:3][CH:4]>>[C:1](=[O:2])O.[O:3][CH:4]",
        "class":      "Esterification",
        "confidence": 0.73,
        "conditions": "DIAD, PPh3, THF",
    },

    # Lactam -> amino acid / amino chain disconnection
    "lactam_opening": {
        "smarts":     "[C:1](=[O:2])[N:3][CH2:4][CH2:5]>>[C:1](=[O:2])O.[N:3][CH2:4][CH2:5]",
        "class":      "Ring Opening",
        "confidence": 0.72,
        "conditions": "H2O, acid or base",
    },
}


# ═══════════════════════════════════════════════════════
#  PURCHASABLE BUILDING BLOCKS
#  Store as raw strings, canonicalize once at import time.
# ═══════════════════════════════════════════════════════

PURCHASABLE_RAW = {
    # Simple aromatics / halides
    "c1ccccc1",            # benzene
    "Clc1ccccc1",          # chlorobenzene
    "BrCc1ccccc1",         # benzyl bromide
    "c1ccncc1",            # pyridine

    # Acids / acyl donors
    "CC(=O)O",             # acetic acid
    "CC(=O)Cl",            # acetyl chloride
    "CC(=O)OC(C)=O",       # acetic anhydride
    "O=C(O)c1ccccc1",      # benzoic acid
    "O=C(O)c1ccc(N)cc1",   # p-aminobenzoic acid
    "O=C(O)c1ccc(O)cc1",   # salicylic acid / hydroxybenzoic acid class

    # Alcohols / phenols
    "CO",                  # methanol
    "CCO",                 # ethanol
    "OCc1ccccc1",          # benzyl alcohol
    "Oc1ccccc1",           # phenol

    # Amines / aldehydes / amino acids
    "Nc1ccccc1",           # aniline
    "c1ccc(N)cc1",         # aniline alt canonical input
    "NCc1ccccc1",          # benzylamine
    "NCC(=O)O",            # glycine
    "O=Cc1ccccc1",         # benzaldehyde

    # Small common reagents / intermediates
    "CC(C)=O",             # acetone
    "CC(=O)Nc1ccccc1",     # acetanilide
}

PURCHASABLE = {
    cs for s in PURCHASABLE_RAW
    if (cs := canon_smiles(s)) is not None
}


def is_purchasable(smiles: str) -> bool:
    """
    Heuristic purchasability check:
    1) Canonicalize first.
    2) Treat very small common molecules as purchasable.
    3) Otherwise match against curated canonical set.
    """
    cs = canon_smiles(smiles)
    if cs is None:
        return False

    mol = safe_mol_from_smiles(cs)
    if mol is None:
        return False

    # Keep this permissive, but not too permissive.
    # Heavy atoms <= 3 covers methanol, ethanol, acetic acid fragments, etc.
    if mol.GetNumHeavyAtoms() <= 3:
        return True

    return cs in PURCHASABLE


# ═══════════════════════════════════════════════════════
#  SYNTHETIC ACCESSIBILITY
# ═══════════════════════════════════════════════════════

def sa_score_approx(smiles: str) -> float:
    """
    Approximate Synthetic Accessibility score (1=easy, 10=hard).
    Simplified heuristic based on MW, rings, stereocenters, rotors.
    For real SA score use: pip install sascorer
    """
    mol = safe_mol_from_smiles(smiles)
    if mol is None:
        return 10.0

    mw = Descriptors.MolWt(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)

    score = 1.0 + (mw / 500.0) * 2.0 + rings * 0.5 + stereo * 0.8 + rot * 0.1
    return min(float(score), 10.0)


# ═══════════════════════════════════════════════════════
#  RDKit ROUTE UTILS
# ═══════════════════════════════════════════════════════

class RDKitRouteUtils:
    """
    RDKit-based route generation and post-processing.
    Used as fallback engine and utilities toolkit.
    """

    def __init__(self):
        self._compiled: Dict[str, Tuple[AllChem.ChemicalReaction, Dict]] = {}
        self._compile_all()

    def _compile_all(self) -> None:
        """Compile all reaction SMARTS once."""
        for tid, tdata in TEMPLATES.items():
            try:
                rxn = AllChem.ReactionFromSmarts(tdata["smarts"])
                if rxn is not None:
                    self._compiled[tid] = (rxn, tdata)
            except Exception:
                continue

    def apply_template(self, smiles: str, tid: str) -> List[List[str]]:
        """
        Apply one retrosynthetic template to a product SMILES.
        Returns up to 3 precursor sets.
        """
        if tid not in self._compiled:
            return []

        mol = safe_mol_from_smiles(smiles)
        if mol is None:
            return []

        rxn, _ = self._compiled[tid]

        try:
            product_sets = rxn.RunReactants((mol,))
        except Exception:
            return []

        results: List[List[str]] = []
        seen_sets: Set[Tuple[str, ...]] = set()
        input_cs = canon_smiles(smiles)

        for ps in product_sets:
            precs: List[str] = []
            valid = True

            for p in ps:
                try:
                    Chem.SanitizeMol(p)
                    s = Chem.MolToSmiles(p, canonical=True)
                except Exception:
                    valid = False
                    break

                if not s:
                    valid = False
                    break

                # Reject identity transform
                if input_cs is not None and s == input_cs:
                    valid = False
                    break

                if not is_reasonable_precursor(s):
                    valid = False
                    break

                precs.append(s)

            if not valid or not precs:
                continue

            precs = dedupe_smiles(precs)
            if not precs:
                continue

            # Prefer multi-precursor disconnections, but allow single-precursor cases
            key = tuple(precs)
            if key in seen_sets:
                continue
            seen_sets.add(key)
            results.append(precs)

        # Keep a few options only
        return results[:3]

    def _step_rank_key(self, step: ReactionStep) -> Tuple[float, int, float]:
        """
        Rank candidate steps by:
        1) precursor purchasable ratio
        2) number of purchasable precursors
        3) confidence
        """
        if not step.precursors:
            return (0.0, 0, step.confidence)

        purch_count = sum(1 for p in step.precursors if is_purchasable(p))
        total = len(step.precursors)
        purch_ratio = purch_count / max(total, 1)

        return (purch_ratio, purch_count, step.confidence)

    def analyze_one(self, smiles: str, depth: int = 0) -> List[ReactionStep]:
        """
        Generate all one-step retrosynthetic disconnections for a molecule.
        """
        steps: List[ReactionStep] = []
        mol = safe_mol_from_smiles(smiles)
        if mol is None:
            return steps

        for tid, (_rxn, tdata) in self._compiled.items():
            prec_sets = self.apply_template(smiles, tid)

            for prec_set in prec_sets:
                purch = all(is_purchasable(p) for p in prec_set)

                step = ReactionStep(
                    smiles=smiles,
                    precursors=prec_set,
                    template_id=tid,
                    template_smarts=tdata["smarts"],
                    reaction_class=tdata["class"],
                    confidence=tdata["confidence"],
                    conditions=tdata["conditions"],
                    depth=depth,
                    is_purchasable=purch,
                    source="rdkit_template",
                )
                steps.append(step)

        # Prefer routes that already land on purchasable precursors
        steps.sort(key=self._step_rank_key, reverse=True)
        return steps

    def run_retrosynthesis(
        self,
        smiles: str,
        max_depth: int = 5,
        max_routes: int = 5,
    ) -> List[SynthesisRoute]:
        """
        Greedy multi-step retrosynthesis.
        Start from different seed templates to diversify route candidates.
        """
        routes: List[SynthesisRoute] = []

        for start_tid in list(self._compiled.keys())[:max_routes]:
            route = self._greedy_route(smiles, start_tid, max_depth)
            if route and route.steps:
                routes.append(route)

        routes.sort(key=lambda r: r.backend_score, reverse=True)
        return routes[:max_routes]

    def _pick_seed_step(
        self,
        smiles: str,
        start_tid: str,
        depth: int,
    ) -> Optional[ReactionStep]:
        """
        Use a specific template as the first step seed if possible.
        This preserves route diversity in run_retrosynthesis().
        """
        if start_tid not in self._compiled:
            return None

        _rxn, tdata = self._compiled[start_tid]
        prec_sets = self.apply_template(smiles, start_tid)

        candidates: List[ReactionStep] = []
        for prec_set in prec_sets:
            purch = all(is_purchasable(p) for p in prec_set)
            candidates.append(
                ReactionStep(
                    smiles=smiles,
                    precursors=prec_set,
                    template_id=start_tid,
                    template_smarts=tdata["smarts"],
                    reaction_class=tdata["class"],
                    confidence=tdata["confidence"],
                    conditions=tdata["conditions"],
                    depth=depth,
                    is_purchasable=purch,
                    source="rdkit_template",
                )
            )

        if not candidates:
            return None

        candidates.sort(key=self._step_rank_key, reverse=True)
        return candidates[0]

    def _greedy_route(
        self,
        smiles: str,
        start_tid: str,
        max_depth: int,
    ) -> SynthesisRoute:
        """
        Greedy route builder:
        - First step tries a chosen seed template
        - Later steps choose best-ranked disconnection
        - Continue only on the largest non-purchasable precursor
        """
        route = SynthesisRoute(
            target_smiles=smiles,
            target_name=smiles[:25],
            backend="rdkit_template",
        )

        current = smiles

        for depth in range(max_depth):
            if depth == 0:
                best = self._pick_seed_step(current, start_tid, depth)
                if best is None:
                    # Fallback to general analysis if the seed template does not apply
                    steps = self.analyze_one(current, depth)
                    if not steps:
                        break
                    best = steps[0]
            else:
                steps = self.analyze_one(current, depth)
                if not steps:
                    break
                best = steps[0]

            route.steps.append(best)

            # Stop if all generated precursors are purchasable
            if best.is_purchasable:
                break

            # Continue only on non-purchasable branches
            non_purch = [p for p in best.precursors if not is_purchasable(p)]
            if not non_purch:
                break

            mols = [(p, safe_mol_from_smiles(p)) for p in non_purch]
            mols = [(s, m) for s, m in mols if m is not None]
            if not mols:
                break

            # Greedy continuation: expand the largest unresolved precursor
            current = max(mols, key=lambda x: x[1].GetNumHeavyAtoms())[0]

        confs = [s.confidence for s in route.steps]
        route.backend_score = float(np.mean(confs)) if confs else 0.0

        # Downstream graph features expected by the rest of the pipeline
        try:
            route.compute_graph_features()
        except Exception:
            pass

        return route

    # ─── MCTS fallback ──────────────────────────────────────────

    def run_mcts_retrosynthesis(
        self,
        smiles: str,
        n_simulations: int = 60,
        max_depth: int = 5,
        n_routes: int = 5,
    ) -> List[SynthesisRoute]:
        """
        MCTS-based retrosynthesis (used as AiZynthFinder fallback).
        """
        from retrodockx.retrosyn.mcts_planner import MCTSPlanner

        planner = MCTSPlanner(
            n_simulations=n_simulations,
            max_depth=max_depth,
            rdkit_utils=self,
        )
        return planner.search(smiles, n_routes=n_routes)
