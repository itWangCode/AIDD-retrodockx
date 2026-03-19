"""
src/retrodockx/ml/shap_utils.py
================================
SHAP-based explainability for all reranker models.

SHAP (SHapley Additive exPlanations) explains *why* a model assigns
a particular score to a route, connecting back to docking, MD, and
synthesis features.

Reference: Lundberg & Lee, NeurIPS 2017
           https://github.com/slundberg/shap
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")
import shap
from typing import List, Dict, Optional, Tuple

from retrodockx.retrosyn.base import SynthesisRoute
from retrodockx.ml.baselines import (
    full_features, route_tabular_features, morgan_fp, rdkit_desc,
    XGBoostReranker, MLPReranker, DualBranchRouteRanker
)


# ─── Feature name registry ──────────────────────────────────────

def get_feature_names(mol_fp_bits: int = 512) -> List[str]:
    """Return full feature name list matching full_features() output."""
    fp_names   = [f"ECFP_bit_{i}" for i in range(mol_fp_bits)]
    desc_names = [
        "MW_norm", "LogP_norm", "TPSA_norm", "HBD_norm", "HBA_norm",
        "NumRings_norm", "RotBonds_norm", "NumAtoms_norm", "ArRings_norm",
        "FrCSP3", "SA_score_norm", "QED_proxy", "Stereocenters_norm",
        "HeavyAtoms_norm",
    ]
    route_names = [
        "route_depth", "step_count", "avg_confidence", "min_confidence",
        "purchasable_ratio", "template_entropy", "branch_factor",
        "rxn_class_diversity", "commercial_bb_ratio", "backend_score",
        "docking_abs_norm", "md_rmsd_norm", "md_stability",
    ]
    return fp_names + desc_names + route_names


def get_route_feature_names() -> List[str]:
    """Feature names for route-only features (Improvement 4)."""
    return [
        "route_depth", "step_count", "avg_confidence", "min_confidence",
        "purchasable_ratio", "template_entropy", "branch_factor",
        "rxn_class_diversity", "commercial_bb_ratio", "backend_score",
        "docking_abs_norm", "md_rmsd_norm", "md_stability",
    ]


# ═══════════════════════════════════════════════════════
#  SHAP EXPLAINER WRAPPERS
# ═══════════════════════════════════════════════════════

class XGBShapExplainer:
    """
    SHAP TreeExplainer for XGBoostReranker.
    Most accurate SHAP method for tree-based models.
    """

    def __init__(self, model: XGBoostReranker):
        self.model = model
        self.explainer = None
        self.feature_names = get_feature_names()

    def build(self, background_routes: List[SynthesisRoute]):
        """Build SHAP explainer using background dataset."""
        X_bg = np.array([full_features(r) for r in background_routes])
        X_bg = self.model.scaler.transform(X_bg)
        self.explainer = shap.TreeExplainer(
            self.model.model,
            data=shap.sample(X_bg, min(50, len(X_bg))),
            feature_names=self.feature_names[:X_bg.shape[1]],
        )
        print(f"  [SHAP-XGB] Explainer built. Background: {len(background_routes)} routes.")

    def explain(self, routes: List[SynthesisRoute]) -> shap.Explanation:
        """Compute SHAP values for a list of routes."""
        if self.explainer is None:
            raise RuntimeError("Call build() first.")
        X = np.array([full_features(r) for r in routes])
        X = self.model.scaler.transform(X)
        sv = self.explainer(X)
        return sv

    def top_features(self, routes: List[SynthesisRoute],
                      n: int = 20) -> List[Tuple[str, float]]:
        """Return top-n features by mean |SHAP value|."""
        sv = self.explain(routes)
        mean_abs = np.abs(sv.values).mean(axis=0)
        names = self.feature_names[:len(mean_abs)]
        ranked = sorted(zip(names, mean_abs), key=lambda x: x[1], reverse=True)
        return ranked[:n]


class MLPShapExplainer:
    """
    SHAP KernelExplainer for MLPReranker.
    Model-agnostic, slower but universal.
    """

    def __init__(self, model: MLPReranker):
        self.model = model
        self.explainer = None
        self.feature_names = get_feature_names()

    def _predict_fn(self, X: np.ndarray) -> np.ndarray:
        """Wrapper to return scalar scores."""
        return self.model.model.predict_proba(X)[:, 1]

    def build(self, background_routes: List[SynthesisRoute]):
        X_bg = np.array([full_features(r) for r in background_routes])
        X_bg = self.model.scaler.transform(X_bg)
        bg_summary = shap.kmeans(X_bg, min(10, len(X_bg)))
        self.explainer = shap.KernelExplainer(self._predict_fn, bg_summary)
        print(f"  [SHAP-MLP] KernelExplainer built (k-means background).")

    def explain(self, routes: List[SynthesisRoute],
                 nsamples: int = 100) -> np.ndarray:
        if self.explainer is None:
            raise RuntimeError("Call build() first.")
        X = np.array([full_features(r) for r in routes])
        X = self.model.scaler.transform(X)
        return self.explainer.shap_values(X, nsamples=nsamples)

    def top_features(self, routes: List[SynthesisRoute],
                      n: int = 20, nsamples: int = 100) -> List[Tuple[str, float]]:
        sv = self.explain(routes, nsamples=nsamples)
        mean_abs = np.abs(sv).mean(axis=0)
        names = self.feature_names[:len(mean_abs)]
        ranked = sorted(zip(names, mean_abs), key=lambda x: x[1], reverse=True)
        return ranked[:n]


class DualBranchShapExplainer:
    """
    SHAP explainer for DualBranchRouteRanker.
    Uses Permutation explainer on route-level features only
    (the most interpretable subset for chemists).
    Explains: why this route was ranked higher than others.
    """

    def __init__(self, model: DualBranchRouteRanker):
        self.model = model
        self.feature_names = get_route_feature_names()
        self.explainer = None

    def _predict_route_only(self, X_route: np.ndarray) -> np.ndarray:
        """Predict using only route features (mol features fixed to mean)."""
        n = X_route.shape[0]
        # Use zeros for mol branch (interpret route contribution)
        X_mol_zero = np.zeros((n, self.model.mol_dim))
        return self.model._forward(X_mol_zero, X_route)

    def build(self, background_routes: List[SynthesisRoute]):
        X_bg = np.array([route_tabular_features(r) for r in background_routes])
        X_bg = self.model.scaler_route.transform(X_bg)
        bg_summary = shap.kmeans(X_bg, min(8, len(X_bg)))
        self.explainer = shap.KernelExplainer(
            self._predict_route_only, bg_summary
        )
        print(f"  [SHAP-DualBranch] Explainer built on route features.")

    def explain(self, routes: List[SynthesisRoute],
                 nsamples: int = 80) -> np.ndarray:
        if self.explainer is None:
            raise RuntimeError("Call build() first.")
        X = np.array([route_tabular_features(r) for r in routes])
        X = self.model.scaler_route.transform(X)
        return self.explainer.shap_values(X, nsamples=nsamples)

    def top_features(self, routes: List[SynthesisRoute],
                      n: int = 13, nsamples: int = 80) -> List[Tuple[str, float]]:
        sv = self.explain(routes, nsamples=nsamples)
        mean_abs = np.abs(sv).mean(axis=0)
        names = self.feature_names[:len(mean_abs)]
        ranked = sorted(zip(names, mean_abs), key=lambda x: x[1], reverse=True)
        return ranked[:n]


# ═══════════════════════════════════════════════════════
#  UNIFIED SHAP RUNNER
# ═══════════════════════════════════════════════════════

def run_shap_analysis(model, routes: List[SynthesisRoute],
                       n_explain: int = 10,
                       nsamples: int = 80) -> Dict:
    """
    Unified SHAP analysis dispatcher.
    Automatically selects the right explainer for the model type.

    Returns dict with SHAP values + top feature ranking.
    """
    print(f"\n  Running SHAP analysis for: {type(model).__name__}")
    print(f"  Routes to explain: {min(n_explain, len(routes))}")

    explain_routes = routes[:n_explain]

    if isinstance(model, XGBoostReranker):
        explainer = XGBShapExplainer(model)
        explainer.build(routes)
        sv = explainer.explain(explain_routes)
        shap_values = sv.values if hasattr(sv, 'values') else np.array(sv)
        base_values = float(sv.base_values.mean()) if hasattr(sv, 'base_values') else 0.0
        top = explainer.top_features(explain_routes, n=20)

    elif isinstance(model, MLPReranker):
        explainer = MLPShapExplainer(model)
        explainer.build(routes)
        shap_values = explainer.explain(explain_routes, nsamples=nsamples)
        base_values = 0.0
        top = explainer.top_features(explain_routes, n=20, nsamples=nsamples)

    elif isinstance(model, DualBranchRouteRanker):
        explainer = DualBranchShapExplainer(model)
        explainer.build(routes)
        shap_values = explainer.explain(explain_routes, nsamples=nsamples)
        base_values = 0.0
        top = explainer.top_features(explain_routes, n=13, nsamples=nsamples)

    else:
        raise TypeError(f"No SHAP explainer for {type(model)}")

    print(f"\n  Top-10 most important features:")
    for rank, (fname, fval) in enumerate(top[:10], 1):
        bar = "█" * int(fval * 40 / (top[0][1] + 1e-9))
        print(f"    {rank:>2}. {fname:<30} {fval:.4f} {bar}")

    return {
        "model_name":   type(model).__name__,
        "shap_values":  shap_values,
        "base_value":   base_values,
        "top_features": top,
        "n_routes":     len(explain_routes),
    }
