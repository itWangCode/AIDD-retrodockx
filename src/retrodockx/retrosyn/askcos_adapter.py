"""
src/retrodockx/retrosyn/askcos_adapter.py
==========================================
ASKCOS backend adapter.

ASKCOS is the MIT Coley group's synthesis planning suite.
- askcos-core: https://github.com/coleygroup/ASKCOS
- Requires: askcos-data (separate download)

This adapter supports two modes:
  1. REST API mode  — calls a running ASKCOS server (local or cloud)
  2. Fallback mode  — uses RDKit templates when ASKCOS is unavailable

The fallback is clearly labeled in route.backend so downstream
models know the source quality.
"""

import os
import json
import requests
import numpy as np
from typing import List, Optional

from retrodockx.retrosyn.base import RetrosynthesisBackend, SynthesisRoute, ReactionStep
from retrodockx.retrosyn.rdkit_route_utils import RDKitRouteUtils


class ASKCOSAdapter(RetrosynthesisBackend):
    """
    Adapter for ASKCOS synthesis planning.

    Priority:
      1. Local askcos-core Python package (if installed)
      2. REST API call to a running ASKCOS server
      3. RDKit template fallback (labeled as "askcos_fallback")

    Reference:
      Coley et al., "A robotic platform for flow synthesis of organic
      compounds informed by AI planning." Science 2019.
      https://github.com/coleygroup/ASKCOS
    """

    ASKCOS_API_ENDPOINTS = {
        "tree_builder":     "/api/tree-builder/",
        "one_step":         "/api/retro/",
        "buyables":         "/api/buyables/",
    }

    def __init__(self,
                 server_url: str = "https://askcos.mit.edu",
                 api_token: str = "",
                 max_depth: int = 5,
                 max_branching: int = 25,
                 expansion_time: int = 30,
                 use_cache: bool = True,
                 cache_dir: str = "data/cache"):
        super().__init__(cache_dir=cache_dir, use_cache=use_cache)
        self.server_url = server_url.rstrip("/")
        self.api_token = api_token
        self.max_depth = max_depth
        self.max_branching = max_branching
        self.expansion_time = expansion_time
        self._rdkit_utils = RDKitRouteUtils()
        self._askcos_available = self._check_availability()

    def name(self) -> str:
        return "askcos"

    def _check_availability(self) -> bool:
        """Check if ASKCOS server is reachable."""
        try:
            r = requests.get(f"{self.server_url}/api/", timeout=5)
            if r.status_code == 200:
                print(f"  [ASKCOS] Server reachable at {self.server_url}")
                return True
        except Exception:
            pass

        # Check local package
        try:
            import askcos  # noqa
            print("  [ASKCOS] Local package found.")
            return True
        except ImportError:
            pass

        print(f"  [ASKCOS] Not available — using RDKit fallback.")
        return False

    def _run_retrosynthesis(self, smiles: str, **kwargs) -> List[SynthesisRoute]:
        if self._askcos_available:
            try:
                return self._run_api(smiles, **kwargs)
            except Exception as e:
                print(f"  [ASKCOS] API error: {e}, falling back to RDKit.")

        return self._run_rdkit_fallback(smiles)

    def _run_api(self, smiles: str, **kwargs) -> List[SynthesisRoute]:
        """Call ASKCOS tree builder REST API."""
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        payload = {
            "smiles": smiles,
            "max_depth": self.max_depth,
            "max_branching": self.max_branching,
            "expansion_time": self.expansion_time,
            "return_first": False,
        }

        url = f"{self.server_url}{self.ASKCOS_API_ENDPOINTS['tree_builder']}"
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        data = response.json()

        routes = []
        for i, tree in enumerate(data.get("trees", [])[:10]):
            route = self._parse_tree(tree, smiles, i)
            if route:
                routes.append(route)

        return routes

    def _parse_tree(self, tree: dict, smiles: str, idx: int) -> Optional[SynthesisRoute]:
        """Parse ASKCOS tree JSON into SynthesisRoute."""
        try:
            steps = self._traverse_tree(tree, depth=0)
            route = SynthesisRoute(
                target_smiles=smiles,
                target_name=tree.get("smiles", smiles)[:30],
                steps=steps,
                backend="askcos",
                backend_score=float(tree.get("plausibility", 0.5)),
            )
            route.compute_graph_features()
            return route
        except Exception as e:
            print(f"    [ASKCOS parse] {e}")
            return None

    def _traverse_tree(self, node: dict, depth: int) -> List[ReactionStep]:
        """Recursively traverse ASKCOS retrosynthetic tree."""
        steps = []
        if "children" not in node or not node["children"]:
            return steps

        for rxn_node in node.get("children", []):
            # Reaction node
            template = rxn_node.get("template_score", {})
            precursors = [c.get("smiles", "") for c in rxn_node.get("children", [])]
            precursors = [p for p in precursors if p]

            step = ReactionStep(
                smiles=node.get("smiles", ""),
                precursors=precursors,
                template_id=rxn_node.get("tforms", [""])[0] if rxn_node.get("tforms") else "",
                template_smarts=rxn_node.get("smiles", ""),
                reaction_class=rxn_node.get("rxn_type", ""),
                confidence=float(rxn_node.get("plausibility", 0.5)),
                depth=depth,
                source="askcos",
            )
            steps.append(step)

            # Recurse into children
            for child in rxn_node.get("children", []):
                steps.extend(self._traverse_tree(child, depth + 1))

        return steps

    def _run_rdkit_fallback(self, smiles: str) -> List[SynthesisRoute]:
        """RDKit template-based fallback, clearly labeled."""
        raw_routes = self._rdkit_utils.run_retrosynthesis(smiles, max_routes=5)
        for r in raw_routes:
            r.backend = "askcos_rdkit_fallback"
        return raw_routes


# ═══════════════════════════════════════════════════════════════
#  AiZynthFinder Adapter
# ═══════════════════════════════════════════════════════════════

class AiZynthFinderAdapter(RetrosynthesisBackend):
    """
    Adapter for AiZynthFinder (Molecular AI, AstraZeneca).

    AiZynthFinder default algorithm: MCTS + neural network template policy.
    - GitHub: https://github.com/MolecularAI/aizynthfinder
    - Paper: Thakkar et al., J. Cheminform. 2020

    Priority:
      1. Local aizynthfinder Python package
      2. RDKit MCTS fallback (labeled as "aizynth_fallback")

    OUR IMPROVEMENTS over vanilla AiZynthFinder:
      - Custom route scoring hook (insert reranker score)
      - Batch mode with shared finder instance
      - Graph feature extraction from route tree
      - Purchasability integration
    """

    def __init__(self,
                 config_path: str = "configs/retrosyn_aizynth.yaml",
                 stock_path: str = "",
                 max_iterations: int = 100,
                 use_cache: bool = True,
                 cache_dir: str = "data/cache"):
        super().__init__(cache_dir=cache_dir, use_cache=use_cache)
        self.config_path = config_path
        self.stock_path = stock_path
        self.max_iterations = max_iterations
        self._finder = None
        self._rdkit_utils = RDKitRouteUtils()
        self._available = self._check_availability()

    def name(self) -> str:
        return "aizynth"

    def _check_availability(self) -> bool:
        try:
            from aizynthfinder.aizynthfinder import AiZynthFinder  # noqa
            print("  [AiZynth] Package found.")
            return True
        except ImportError:
            print("  [AiZynth] Not installed — using MCTS+RDKit fallback.")
            return False

    def _get_finder(self):
        """Lazy-load AiZynthFinder instance (shared across calls for speed)."""
        if self._finder is None and self._available:
            from aizynthfinder.aizynthfinder import AiZynthFinder
            self._finder = AiZynthFinder(configfile=self.config_path)
            if self.stock_path and os.path.exists(self.stock_path):
                self._finder.stock.select(self.stock_path)
        return self._finder

    def _run_retrosynthesis(self, smiles: str, **kwargs) -> List[SynthesisRoute]:
        if self._available:
            try:
                return self._run_local(smiles, **kwargs)
            except Exception as e:
                print(f"  [AiZynth] Error: {e}, falling back.")

        return self._run_mcts_fallback(smiles)

    def _run_local(self, smiles: str, **kwargs) -> List[SynthesisRoute]:
        """Run AiZynthFinder locally."""
        finder = self._get_finder()
        finder.target_smiles = smiles
        finder.config.iteration_limit = self.max_iterations
        finder.tree_analysis()

        routes = []
        for i, route_dict in enumerate(finder.routes.top_routes()):
            route = self._parse_aizynth_route(route_dict, smiles, i)
            if route:
                routes.append(route)

        return routes

    def _parse_aizynth_route(self, route_dict: dict, smiles: str,
                              idx: int) -> Optional[SynthesisRoute]:
        """Convert AiZynthFinder route dict to SynthesisRoute."""
        try:
            # AiZynthFinder route dict structure:
            # {'smiles': ..., 'type': 'mol'|'reaction', 'children': [...]}
            steps = self._extract_steps(route_dict, depth=0)
            score = float(route_dict.get("score", 0.5))
            route = SynthesisRoute(
                target_smiles=smiles,
                target_name=f"AiZynth_Route_{idx+1}",
                steps=steps,
                backend="aizynth",
                backend_score=score,
            )
            route.compute_graph_features()
            return route
        except Exception as e:
            print(f"    [AiZynth parse] {e}")
            return None

    def _extract_steps(self, node: dict, depth: int) -> List[ReactionStep]:
        """Recursively extract steps from AiZynthFinder route tree."""
        steps = []
        if node.get("type") == "reaction":
            precursors = [c.get("smiles", "") for c in node.get("children", [])
                          if c.get("type") == "mol"]
            parent_smiles = node.get("in_stock", "")

            step = ReactionStep(
                smiles=node.get("smiles", ""),
                precursors=[p for p in precursors if p],
                template_id=node.get("template_id", ""),
                template_smarts=node.get("reaction_smarts", ""),
                reaction_class=node.get("classification", ""),
                confidence=float(node.get("policy_probability", 0.5)),
                depth=depth,
                is_purchasable=all(
                    c.get("in_stock", False)
                    for c in node.get("children", [])
                    if c.get("type") == "mol"
                ),
                source="aizynth",
            )
            steps.append(step)

        for child in node.get("children", []):
            steps.extend(self._extract_steps(child, depth + 1))

        return steps

    def _run_mcts_fallback(self, smiles: str) -> List[SynthesisRoute]:
        """MCTS + RDKit template fallback."""
        routes = self._rdkit_utils.run_mcts_retrosynthesis(
            smiles, n_simulations=60, max_depth=5, n_routes=5
        )
        for r in routes:
            r.backend = "aizynth_mcts_fallback"
        return routes
