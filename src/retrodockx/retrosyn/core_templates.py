"""
src/retrodockx/retrosyn/core_templates.py
=========================================
Curated core retrosynthetic reaction templates.

Design goals
------------
1. Keep this file focused on a compact, high-value template library.
2. Templates are hand-curated and intended for fallback / interpretable routing.
3. Each template entry contains:
   - smarts: retrosynthetic reaction SMARTS
   - class: coarse reaction family label
   - confidence: prior confidence for fallback ranking
   - conditions: human-readable representative conditions
   - literature_tag: high-level provenance / reaction family
   - scope_note: brief intended use / limitations

Notes
-----
- These are NOT meant to exhaustively cover organic chemistry.
- They are intentionally conservative starter templates for route planning.
- RDKit reaction SMARTS are used in retrosynthetic direction:
      target >> precursors
"""

from __future__ import annotations

from typing import Dict, Any


TEMPLATES: Dict[str, Dict[str, Any]] = {
    # ─────────────────────────────────────────────
    # Acyl derivatives / hydrolysis / acylation
    # ─────────────────────────────────────────────
    "ester_hydrolysis": {
        "smarts": "[C:1](=[O:2])[O:3][C,c:4]>>[C:1](=[O:2])O.[O:3][C,c:4]",
        "class": "Hydrolysis",
        "confidence": 0.90,
        "conditions": "H2O, H+ or OH-, heat",
        "literature_tag": "ester cleavage / medicinal chemistry common transformation",
        "scope_note": "Generic ester disconnection to carboxylic acid + alcohol/phenol precursor.",
    },
    "aryl_ester_hydrolysis": {
        "smarts": "[c:4][O:3][C:1](=[O:2])[C:5]>>[c:4][O:3].[C:5][C:1](=[O:2])O",
        "class": "Hydrolysis",
        "confidence": 0.86,
        "conditions": "Aqueous acid/base, heat",
        "literature_tag": "aryl ester cleavage",
        "scope_note": "Useful for acetate-like aryl ester disconnections.",
    },
    "amide_hydrolysis": {
        "smarts": "[C:1](=[O:2])[N:3][C,c:4]>>[C:1](=[O:2])O.[N:3][C,c:4]",
        "class": "Hydrolysis",
        "confidence": 0.85,
        "conditions": "H2O, H+ or OH-, heat",
        "literature_tag": "amide cleavage",
        "scope_note": "Generic amide disconnection to acid + amine precursor.",
    },
    "amide_hydrolysis_secondary": {
        "smarts": "[C:1](=[O:2])[N:3]([C,c:4])[C,c:5]>>[C:1](=[O:2])O.[N:3]([C,c:4])[C,c:5]",
        "class": "Hydrolysis",
        "confidence": 0.82,
        "conditions": "Strong aqueous acid/base, heat",
        "literature_tag": "tertiary amide / secondary amine disconnection",
        "scope_note": "For more substituted amide motifs.",
    },
    "n_acylation_aryl_or_alkyl_amine": {
        "smarts": "[N:1][C:2](=[O:3])[C,c:4]>>[N:1].[C,c:4][C:2](=[O:3])Cl",
        "class": "Acylation",
        "confidence": 0.88,
        "conditions": "Et3N, DCM, 0C to RT",
        "literature_tag": "acid chloride acylation",
        "scope_note": "Generic N-acylation disconnection to amine + acyl chloride.",
    },
    "amide_coupling_direct": {
        "smarts": "[C:1](=[O:2])[N:3][C,c:4]>>[C:1](=[O:2])O.[N:3][C,c:4]",
        "class": "Amide Coupling",
        "confidence": 0.84,
        "conditions": "EDC/HATU/DIC coupling conditions",
        "literature_tag": "amide coupling",
        "scope_note": "Alternative amide disconnection to free acid + amine.",
    },
    "sulfonamide_formation": {
        "smarts": "[S:1](=[O:2])(=[O:3])[N:4][C,c:5]>>[S:1](=[O:2])(=[O:3])Cl.[N:4][C,c:5]",
        "class": "Sulfonylation",
        "confidence": 0.89,
        "conditions": "Base, DCM or pyridine",
        "literature_tag": "sulfonamide formation",
        "scope_note": "Sulfonamide disconnection to sulfonyl chloride + amine.",
    },
    "carbamate_cleavage": {
        "smarts": "[O,N:1][C:2](=[O:3])[O,N:4][C,c:5]>>[O,N:1][C:2](=[O:3])Cl.[O,N:4][C,c:5]",
        "class": "Carbamate Formation",
        "confidence": 0.76,
        "conditions": "Base, low temperature",
        "literature_tag": "carbamate / chloroformate chemistry",
        "scope_note": "Generic carbamate/urethane-type disconnection.",
    },
    "urea_disconnection": {
        "smarts": "[N:1][C:2](=[O:3])[N:4][C,c:5]>>[N:1]C#N.[N:4][C,c:5]",
        "class": "Urea Formation",
        "confidence": 0.70,
        "conditions": "Isocyanate / carbamoyl equivalent formation",
        "literature_tag": "urea formation",
        "scope_note": "Heuristic urea disconnection; fallback only.",
    },

    # ─────────────────────────────────────────────
    # Protection / deprotection
    # ─────────────────────────────────────────────
    "acetylation_deprotection": {
        "smarts": "[C,c:1][O:2][C:3](=[O:4])C>>[C,c:1][OH:2]",
        "class": "Deprotection",
        "confidence": 0.88,
        "conditions": "Methanolysis / aqueous base",
        "literature_tag": "acetate deprotection",
        "scope_note": "ArOAc / ROAc cleavage to alcohol or phenol.",
    },
    "boc_deprotection": {
        "smarts": "[N:1][C:2](=[O:3])O[C:4](C)(C)C>>[N:1]",
        "class": "Deprotection",
        "confidence": 0.86,
        "conditions": "TFA, DCM",
        "literature_tag": "Boc deprotection",
        "scope_note": "Simplified Boc removal template.",
    },
    "cbz_deprotection": {
        "smarts": "[N:1][C:2](=[O:3])O[CH2:4][c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1>>[N:1]",
        "class": "Deprotection",
        "confidence": 0.74,
        "conditions": "H2, Pd/C",
        "literature_tag": "Cbz deprotection",
        "scope_note": "Simplified Cbz removal fallback template.",
    },
    "benzyl_ether_deprotection": {
        "smarts": "[O:1][CH2:2][c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1>>[O:1]",
        "class": "Deprotection",
        "confidence": 0.78,
        "conditions": "H2, Pd/C or BCl3",
        "literature_tag": "benzyl ether deprotection",
        "scope_note": "Generic O-benzyl cleavage fallback.",
    },
    "methyl_ester_cleavage": {
        "smarts": "[C:1](=[O:2])OC>>[C:1](=[O:2])O.CO",
        "class": "Hydrolysis",
        "confidence": 0.87,
        "conditions": "NaOH, H2O/MeOH",
        "literature_tag": "methyl ester hydrolysis",
        "scope_note": "Especially helpful for methyl ester substrates.",
    },

    # ─────────────────────────────────────────────
    # Ether / alkylation / substitution
    # ─────────────────────────────────────────────
    "o_alkylation_benzyl": {
        "smarts": "[C,c:1][O:2][CH2:3][C,c:4]>>[C,c:1]O.[Br][CH2:3][C,c:4]",
        "class": "Etherification",
        "confidence": 0.80,
        "conditions": "K2CO3, DMF, 60C",
        "literature_tag": "Williamson / benzylation",
        "scope_note": "Generic O-benzyl / O-alkyl disconnection.",
    },
    "n_alkylation": {
        "smarts": "[N:1][CH2:2][C,c:3]>>[N:1].[Br][CH2:2][C,c:3]",
        "class": "N-Alkylation",
        "confidence": 0.79,
        "conditions": "Base, DMF or MeCN",
        "literature_tag": "alkyl halide substitution",
        "scope_note": "Simple amine alkylation disconnection.",
    },
    "sn2_alkylation_heteroatom": {
        "smarts": "[O,N,S:1][CH2:2][C,c:3]>>[O,N,S:1].[Br][CH2:2][C,c:3]",
        "class": "Substitution",
        "confidence": 0.76,
        "conditions": "Base, polar aprotic solvent",
        "literature_tag": "SN2 heteroatom alkylation",
        "scope_note": "Fallback for O/N/S alkyl substitutions.",
    },
    "ether_cleavage_aryl_alkyl": {
        "smarts": "[c:1][O:2][C:3]>>[c:1][OH:2].[C:3]Br",
        "class": "Etherification",
        "confidence": 0.72,
        "conditions": "Williamson ether synthesis logic",
        "literature_tag": "aryl-alkyl ether disconnection",
        "scope_note": "Useful fallback for aryl alkyl ethers.",
    },

    # ─────────────────────────────────────────────
    # C-N / carbonyl chemistry
    # ─────────────────────────────────────────────
    "reductive_amination": {
        "smarts": "[C,c:1][NH:2][CH2:3][C,c:4]>>[C,c:1][NH2:2].[C,c:4][CH:3]=O",
        "class": "Amination",
        "confidence": 0.78,
        "conditions": "NaBH3CN, MeOH",
        "literature_tag": "reductive amination",
        "scope_note": "Secondary amine disconnection to amine + aldehyde/ketone equivalent.",
    },
    "imine_reduction_reverse": {
        "smarts": "[N:1][CH2:2][C,c:3]>>[N:1].[C,c:3][CH:2]=O",
        "class": "Amination",
        "confidence": 0.72,
        "conditions": "Reductive amination logic",
        "literature_tag": "imine/reductive amination",
        "scope_note": "Simplified fallback for benzylic amino motifs.",
    },
    "aniline_from_nitro": {
        "smarts": "[N:1][c:2]>>O=[N+]([O-])[c:2]",
        "class": "Reduction",
        "confidence": 0.68,
        "conditions": "H2, Pd/C or Fe/HCl",
        "literature_tag": "nitro reduction",
        "scope_note": "Retrosynthetic disconnection of aniline-like motifs to nitro precursor.",
    },

    # ─────────────────────────────────────────────
    # Oxidation / reduction interconversions
    # ─────────────────────────────────────────────
    "oxidation_primary_alcohol_to_acid": {
        "smarts": "[C,c:1][C:2](=[O:3])O>>[C,c:1][CH2:2]O",
        "class": "Oxidation",
        "confidence": 0.75,
        "conditions": "TEMPO/NaOCl or PCC then further oxidation",
        "literature_tag": "alcohol oxidation",
        "scope_note": "Acid disconnection to primary alcohol precursor.",
    },
    "oxidation_aldehyde_from_alcohol": {
        "smarts": "[C,c:1][CH:2]=O>>[C,c:1][CH2:2]O",
        "class": "Oxidation",
        "confidence": 0.73,
        "conditions": "PCC, Dess-Martin, TEMPO",
        "literature_tag": "alcohol oxidation",
        "scope_note": "Aldehyde disconnection to alcohol precursor.",
    },
    "ketone_reduction_reverse": {
        "smarts": "[C,c:1][CH:2](O)[C,c:3]>>[C,c:1][C:2](=O)[C,c:3]",
        "class": "Reduction",
        "confidence": 0.70,
        "conditions": "NaBH4 / LiAlH4 forward logic",
        "literature_tag": "ketone reduction",
        "scope_note": "Alcohol disconnection to ketone precursor.",
    },

    # ─────────────────────────────────────────────
    # Cross-couplings / aromatic substitutions
    # ─────────────────────────────────────────────
    "suzuki_biaryl": {
        "smarts": "[c:1]-[c:2]>>[c:1]Br.[c:2]B(O)O",
        "class": "C-C Coupling",
        "confidence": 0.85,
        "conditions": "Pd(PPh3)4, K2CO3, EtOH/H2O",
        "literature_tag": "Suzuki-Miyaura coupling",
        "scope_note": "Generic biaryl disconnection to aryl bromide + boronic acid.",
    },
    "buchwald_hartwig_cn": {
        "smarts": "[c:1][N:2][C,c:3]>>[c:1]Br.[N:2][C,c:3]",
        "class": "C-N Coupling",
        "confidence": 0.82,
        "conditions": "Pd catalyst, ligand, base",
        "literature_tag": "Buchwald-Hartwig amination",
        "scope_note": "Aryl amine disconnection to aryl bromide + amine.",
    },
    "ullmann_cn": {
        "smarts": "[c:1][N:2][C,c:3]>>[c:1]I.[N:2][C,c:3]",
        "class": "C-N Coupling",
        "confidence": 0.70,
        "conditions": "Cu catalyst, base, heat",
        "literature_tag": "Ullmann C-N coupling",
        "scope_note": "Fallback aromatic C-N bond disconnection.",
    },
    "ullmann_co": {
        "smarts": "[c:1][O:2][C,c:3]>>[c:1]I.[O:2][C,c:3]",
        "class": "C-O Coupling",
        "confidence": 0.68,
        "conditions": "Cu catalyst, base, heat",
        "literature_tag": "Ullmann C-O coupling",
        "scope_note": "Fallback aromatic C-O bond disconnection.",
    },
    "snar_aryl_amine": {
        "smarts": "[c:1][N:2][C,c:3]>>[c:1]Cl.[N:2][C,c:3]",
        "class": "SNAr",
        "confidence": 0.80,
        "conditions": "Base, polar solvent, heat",
        "literature_tag": "nucleophilic aromatic substitution",
        "scope_note": "Useful for electron-poor aryl amines.",
    },
    "snar_aryl_ether": {
        "smarts": "[c:1][O:2][C,c:3]>>[c:1]F.[O:2][C,c:3]",
        "class": "SNAr",
        "confidence": 0.76,
        "conditions": "Base, DMF/DMSO, heat",
        "literature_tag": "nucleophilic aromatic substitution",
        "scope_note": "Fallback for aryl ether formation via SNAr.",
    },
    "sonogashira": {
        "smarts": "[c:1][C]#[C:2][C,c:3]>>[c:1]I.[C]#[C:2][C,c:3]",
        "class": "C-C Coupling",
        "confidence": 0.72,
        "conditions": "Pd/Cu catalyst, amine base",
        "literature_tag": "Sonogashira coupling",
        "scope_note": "Simplified aryl-alkyne disconnection.",
    },
    "heck_alkenylation": {
        "smarts": "[c:1][CH:2]=[CH:3][C,c:4]>>[c:1]Br.[CH2:2]=[CH:3][C,c:4]",
        "class": "C-C Coupling",
        "confidence": 0.66,
        "conditions": "Pd catalyst, base, heat",
        "literature_tag": "Heck coupling",
        "scope_note": "Fallback styrenyl disconnection.",
    },

    # ─────────────────────────────────────────────
    # Ring / lactam / lactone openings
    # ─────────────────────────────────────────────
    "lactam_opening": {
        "smarts": "[C:1](=[O:2])[N:3][CH2:4][CH2:5]>>[C:1](=[O:2])O.[N:3][CH2:4][CH2:5]",
        "class": "Ring Opening",
        "confidence": 0.72,
        "conditions": "H2O, acid or base",
        "literature_tag": "lactam opening",
        "scope_note": "Fallback ring-opening disconnection.",
    },
    "lactone_opening": {
        "smarts": "[C:1](=[O:2])O[CH2:3][CH2:4]>>[C:1](=[O:2])O.O[CH2:3][CH2:4]",
        "class": "Ring Opening",
        "confidence": 0.70,
        "conditions": "Aqueous acid/base",
        "literature_tag": "lactone opening",
        "scope_note": "Fallback lactone cleavage.",
    },
    "mitsunobu_ester": {
        "smarts": "[C:1](=[O:2])[O:3][CH:4]>>[C:1](=[O:2])O.[O:3][CH:4]",
        "class": "Esterification",
        "confidence": 0.73,
        "conditions": "DIAD, PPh3, THF",
        "literature_tag": "Mitsunobu / esterification logic",
        "scope_note": "Secondary alcohol ester disconnection.",
    },
}


def get_core_templates() -> Dict[str, Dict[str, Any]]:
    """Return a copy of the curated template dictionary."""
    return dict(TEMPLATES)
