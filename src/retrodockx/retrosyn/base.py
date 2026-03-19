"""
src/retrodockx/retrosyn/base.py
================================
Improvement 1: Unified multi-backend retrosynthesis interface.

Provides a single RetrosynthesisBackend ABC that both ASKCOS and
AiZynthFinder adapters implement, so the rest of the pipeline never
needs to know which engine is running.

Design pattern: Adapter + Strategy
"""

import abc
import time
import json
import hashlib
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


# ═══════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════

@dataclass
class ReactionStep:
    """A single retrosynthetic disconnection step."""
    smiles: str                      # Molecule being disconnected
    precursors: List[str]            # Resulting precursors
    template_id: str = ""
    template_smarts: str = ""
    reaction_class: str = ""
    confidence: float = 0.0
    conditions: str = ""
    depth: int = 0
    is_purchasable: bool = False
    source: str = ""                 # "askcos" | "aizynth" | "rdkit"

    # Improvement 4: route feasibility graph features
    precursor_count: int = 0
    template_entropy: float = 0.0   # uncertainty over template distribution
    branch_depth: int = 0
    purchasable_ratio: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def __post_init__(self):
        self.precursor_count = len(self.precursors)
        if self.precursor_count > 0:
            self.purchasable_ratio = 1.0 if self.is_purchasable else 0.0


@dataclass
class SynthesisRoute:
    """
    A complete multi-step synthesis route.
    Contains all data needed by the reranker.
    """
    target_smiles: str
    target_name: str
    steps: List[ReactionStep] = field(default_factory=list)
    backend: str = "unknown"

    # Route-level scores (from backend)
    backend_score: float = 0.0

    # Improvement 4: Route feasibility graph features
    route_depth: int = 0
    step_count: int = 0
    avg_confidence: float = 0.0
    min_confidence: float = 0.0
    purchasable_ratio: float = 0.0
    template_entropy: float = 0.0    # avg entropy over steps
    branch_factor: float = 0.0       # avg precursors per step
    reaction_class_diversity: float = 0.0
    commercial_bb_ratio: float = 0.0

    # Docking/MD context (filled later by pipeline)
    docking_score: float = 0.0
    md_rmsd: float = 0.0
    md_stability: float = 0.0

    # Final reranker score (filled by ml module)
    reranker_score: float = 0.0
    rank: int = 0

    def compute_graph_features(self):
        """Compute Improvement 4 route graph features from steps."""
        import numpy as np
        if not self.steps:
            return

        self.step_count = len(self.steps)
        self.route_depth = max((s.depth for s in self.steps), default=0) + 1
        confs = [s.confidence for s in self.steps]
        self.avg_confidence = float(np.mean(confs)) if confs else 0.0
        self.min_confidence = float(np.min(confs)) if confs else 0.0
        self.purchasable_ratio = float(
            np.mean([s.purchasable_ratio for s in self.steps])
        )
        # Template entropy: -sum(p log p) over confidence distribution
        probs = np.array(confs)
        probs = probs / (probs.sum() + 1e-9)
        self.template_entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
        self.branch_factor = float(np.mean([s.precursor_count for s in self.steps]))

        # Reaction class diversity (unique categories / total)
        classes = [s.reaction_class for s in self.steps if s.reaction_class]
        self.reaction_class_diversity = (
            len(set(classes)) / len(classes) if classes else 0.0
        )

    def feature_vector(self) -> Dict[str, float]:
        """Return flat feature dict for ML model."""
        self.compute_graph_features()
        return {
            "route_depth":                self.route_depth,
            "step_count":                 self.step_count,
            "avg_confidence":             self.avg_confidence,
            "min_confidence":             self.min_confidence,
            "purchasable_ratio":          self.purchasable_ratio,
            "template_entropy":           self.template_entropy,
            "branch_factor":              self.branch_factor,
            "reaction_class_diversity":   self.reaction_class_diversity,
            "commercial_bb_ratio":        self.commercial_bb_ratio,
            "backend_score":              self.backend_score,
            "docking_score":              self.docking_score,
            "md_rmsd":                    self.md_rmsd,
            "md_stability":               self.md_stability,
        }

    def to_dict(self) -> dict:
        d = asdict(self)
        d["feature_vector"] = self.feature_vector()
        return d


# ═══════════════════════════════════════════════════════
#  ABSTRACT BASE
# ═══════════════════════════════════════════════════════

class RetrosynthesisBackend(abc.ABC):
    """
    Abstract base class for all retrosynthesis backends.
    Both ASKCOS and AiZynthFinder implement this interface.
    """

    def __init__(self, cache_dir: str = "data/cache", use_cache: bool = True):
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        os.makedirs(cache_dir, exist_ok=True)

    @abc.abstractmethod
    def name(self) -> str:
        """Backend identifier string."""

    @abc.abstractmethod
    def _run_retrosynthesis(self, smiles: str, **kwargs) -> List[SynthesisRoute]:
        """Backend-specific implementation. Returns list of candidate routes."""

    def _cache_key(self, smiles: str) -> str:
        return hashlib.md5(f"{self.name()}:{smiles}".encode()).hexdigest()[:16]

    def _cache_path(self, smiles: str) -> str:
        return os.path.join(self.cache_dir, f"{self._cache_key(smiles)}.json")

    def _load_cache(self, smiles: str) -> Optional[List[dict]]:
        path = self._cache_path(smiles)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    def _save_cache(self, smiles: str, routes: List[SynthesisRoute]):
        path = self._cache_path(smiles)
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in routes], f, indent=2)

    def plan(self, smiles: str, max_retries: int = 3,
             **kwargs) -> List[SynthesisRoute]:
        """
        Public entry point. Handles caching + retry.
        Improvement 5: auto-cache + auto-retry logic.
        """
        # Cache hit
        if self.use_cache:
            cached = self._load_cache(smiles)
            if cached is not None:
                print(f"  [CACHE] {self.name()} hit for {smiles[:30]}...")
                return self._deserialize_routes(cached)

        # Run with retry
        for attempt in range(1, max_retries + 1):
            try:
                routes = self._run_retrosynthesis(smiles, **kwargs)
                if self.use_cache:
                    self._save_cache(smiles, routes)
                return routes
            except Exception as e:
                print(f"  [RETRY {attempt}/{max_retries}] {self.name()}: {e}")
                if attempt < max_retries:
                    time.sleep(1.5 ** attempt)

        print(f"  [FAIL] {self.name()} failed after {max_retries} retries.")
        return []

    def plan_batch(self, smiles_list: List[str],
                   names: Optional[List[str]] = None,
                   **kwargs) -> Dict[str, List[SynthesisRoute]]:
        """
        Improvement 5: batch pipeline sharing the same backend.
        Returns {smiles: [routes]}.
        """
        if names is None:
            names = [f"Mol_{i+1}" for i in range(len(smiles_list))]

        results = {}
        total = len(smiles_list)
        print(f"\n{'='*58}")
        print(f"  [{self.name()}] Batch retrosynthesis: {total} molecules")
        print(f"{'='*58}")

        for i, (smiles, name) in enumerate(zip(smiles_list, names), 1):
            print(f"  [{i:>3}/{total}] {name} — {smiles[:35]}...")
            routes = self.plan(smiles, **kwargs)
            results[smiles] = routes
            print(f"          → {len(routes)} routes found")

        print(f"\n  Done. {sum(len(v) for v in results.values())} total routes.")
        return results

    @staticmethod
    def _deserialize_routes(data: List[dict]) -> List[SynthesisRoute]:
        routes = []
        for d in data:
            steps = [ReactionStep(**s) for s in d.get("steps", [])]
            r = SynthesisRoute(
                target_smiles=d["target_smiles"],
                target_name=d["target_name"],
                steps=steps,
                backend=d.get("backend", ""),
                backend_score=d.get("backend_score", 0.0),
                docking_score=d.get("docking_score", 0.0),
                md_rmsd=d.get("md_rmsd", 0.0),
                md_stability=d.get("md_stability", 0.0),
            )
            routes.append(r)
        return routes
