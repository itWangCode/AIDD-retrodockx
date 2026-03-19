"""
src/retrodockx/retrosyn/template_loader.py
==========================================
Template loading / validation utilities.

Design goals
------------
1. Keep route utilities independent from template source.
2. Support:
   - built-in curated core templates
   - optional JSON / YAML extension files
3. Validate fields and normalize template metadata.
4. Provide a single loader function for rdkit_route_utils.py

Usage
-----
from .core_templates import get_core_templates
templates = load_templates()

Optional file formats
---------------------
JSON:
{
  "template_id": {
    "smarts": "...",
    "class": "...",
    "confidence": 0.8,
    "conditions": "...",
    "literature_tag": "...",
    "scope_note": "..."
  }
}

YAML:
template_id:
  smarts: "..."
  class: "..."
  confidence: 0.8
  conditions: "..."
  literature_tag: "..."
  scope_note: "..."
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from rdkit.Chem import AllChem

from retrodockx.retrosyn.core_templates import get_core_templates


REQUIRED_TEMPLATE_KEYS = {
    "smarts",
    "class",
    "confidence",
    "conditions",
}

OPTIONAL_TEMPLATE_KEYS = {
    "literature_tag",
    "scope_note",
    "priority",
}


def _try_import_yaml():
    try:
        import yaml  # type: ignore
        return yaml
    except Exception:
        return None


def normalize_template_entry(template_id: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a single template entry and fill defaults.
    """
    out = dict(entry)

    missing = [k for k in REQUIRED_TEMPLATE_KEYS if k not in out]
    if missing:
        raise ValueError(
            f"Template '{template_id}' is missing required keys: {missing}"
        )

    out["smarts"] = str(out["smarts"]).strip()
    out["class"] = str(out["class"]).strip()
    out["conditions"] = str(out["conditions"]).strip()
    out["confidence"] = float(out["confidence"])

    if not (0.0 <= out["confidence"] <= 1.0):
        raise ValueError(
            f"Template '{template_id}' has invalid confidence: {out['confidence']}"
        )

    out.setdefault("literature_tag", "curated / unspecified")
    out.setdefault("scope_note", "")
    out.setdefault("priority", 0)

    return out


def validate_reaction_smarts(template_id: str, smarts: str) -> None:
    """
    Validate RDKit reaction SMARTS by attempting compilation.
    """
    try:
        rxn = AllChem.ReactionFromSmarts(smarts)
        if rxn is None:
            raise ValueError(f"RDKit returned None for template '{template_id}'")
    except Exception as e:
        raise ValueError(
            f"Template '{template_id}' has invalid reaction SMARTS: {smarts}\n{e}"
        ) from e


def validate_templates(templates: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Validate and normalize a template dictionary.
    """
    validated: Dict[str, Dict[str, Any]] = {}
    for tid, entry in templates.items():
        norm = normalize_template_entry(tid, entry)
        validate_reaction_smarts(tid, norm["smarts"])
        validated[tid] = norm
    return validated


def load_templates_from_json(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load template dictionary from a JSON file.
    """
    if not path or not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"JSON template file must contain an object/dict: {path}")

    out: Dict[str, Dict[str, Any]] = {}
    for tid, entry in data.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Template '{tid}' in {path} is not a dict")
        out[str(tid)] = dict(entry)
    return out


def load_templates_from_yaml(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load template dictionary from a YAML file.
    """
    if not path or not os.path.exists(path):
        return {}

    yaml = _try_import_yaml()
    if yaml is None:
        raise ImportError(
            "PyYAML is not installed, cannot load YAML template file."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"YAML template file must contain a dict at top level: {path}")

    out: Dict[str, Dict[str, Any]] = {}
    for tid, entry in data.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Template '{tid}' in {path} is not a dict")
        out[str(tid)] = dict(entry)
    return out


def load_templates_from_file(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load templates from a .json / .yaml / .yml file.
    """
    if not path:
        return {}

    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        return load_templates_from_json(path)
    if ext in {".yaml", ".yml"}:
        return load_templates_from_yaml(path)

    raise ValueError(f"Unsupported template file extension: {path}")


def merge_template_dicts(
    base: Dict[str, Dict[str, Any]],
    extra: Dict[str, Dict[str, Any]],
    overwrite: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Merge two template dictionaries.
    If overwrite=True, extra entries override base entries with same id.
    """
    merged = dict(base)
    for tid, entry in extra.items():
        if overwrite or tid not in merged:
            merged[tid] = dict(entry)
    return merged


def load_templates(
    extra_template_file: Optional[str] = None,
    overwrite: bool = True,
    validate: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Load templates for retrosynthesis.

    Steps:
    1) load built-in curated templates
    2) optionally merge external JSON/YAML template file
    3) optionally validate all entries

    Returns
    -------
    dict
        Template dictionary ready for RDKit compilation.
    """
    templates = get_core_templates()

    if extra_template_file:
        extra = load_templates_from_file(extra_template_file)
        templates = merge_template_dicts(templates, extra, overwrite=overwrite)

    if validate:
        templates = validate_templates(templates)

    return templates
