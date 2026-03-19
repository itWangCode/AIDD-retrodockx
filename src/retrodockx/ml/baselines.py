"""
src/retrodockx/ml/baselines.py + dual_branch_model.py (combined)
=================================================================
Improvement 2: Route Reranker Models
  - XGBoostReranker        (baseline)
  - MLPReranker            (baseline)
  - DualBranchRouteRanker  (our improved model)

Improvement 3: Multi-objective loss
  L = ranking_loss
    + α * synthesizability_loss
    + β * docking_consistency_loss
    + γ * md_stability_loss

Improvement 4: Route feasibility graph features used as input to all models.
"""

import os
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from typing import List, Dict, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ndcg_score
import xgboost as xgb

from retrodockx.retrosyn.base import SynthesisRoute
from retrodockx.retrosyn.rdkit_route_utils import sa_score_approx


# ═══════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════

def morgan_fp(smiles: str, nbits: int = 1024) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nbits)
    return np.array(GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits), dtype=float)


def rdkit_desc(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(14)
    try:
        return np.array([
            Descriptors.MolWt(mol) / 600,
            Descriptors.MolLogP(mol) / 8,
            Descriptors.TPSA(mol) / 200,
            rdMolDescriptors.CalcNumHBD(mol) / 10,
            rdMolDescriptors.CalcNumHBA(mol) / 15,
            rdMolDescriptors.CalcNumRings(mol) / 8,
            rdMolDescriptors.CalcNumRotatableBonds(mol) / 15,
            mol.GetNumHeavyAtoms() / 80,
            rdMolDescriptors.CalcNumAromaticRings(mol) / 6,
            rdMolDescriptors.CalcFractionCSP3(mol),
            sa_score_approx(smiles) / 10,
            Descriptors.MolWt(mol) / 600,    # QED proxy
            rdMolDescriptors.CalcNumStereocenters(mol) / 5,
            rdMolDescriptors.CalcNumHeavyAtoms(mol) / 80,
        ], dtype=float)
    except Exception:
        return np.zeros(14)


def route_tabular_features(route: SynthesisRoute) -> np.ndarray:
    """Improvement 4: Extract route graph features as numpy array."""
    fv = route.feature_vector()
    return np.array([
        fv.get("route_depth", 0) / 8,
        fv.get("step_count", 0) / 8,
        fv.get("avg_confidence", 0),
        fv.get("min_confidence", 0),
        fv.get("purchasable_ratio", 0),
        fv.get("template_entropy", 0) / 3,
        fv.get("branch_factor", 0) / 4,
        fv.get("reaction_class_diversity", 0),
        fv.get("commercial_bb_ratio", 0),
        fv.get("backend_score", 0),
        min(abs(fv.get("docking_score", 0)) / 15, 1.0),   # docking: better = more negative
        fv.get("md_rmsd", 0) / 10,
        fv.get("md_stability", 0),
    ], dtype=float)


def full_features(route: SynthesisRoute) -> np.ndarray:
    """Combined molecule + route features for tabular models (XGB, MLP)."""
    mol_fp   = morgan_fp(route.target_smiles, nbits=512)
    mol_desc = rdkit_desc(route.target_smiles)
    route_f  = route_tabular_features(route)
    return np.concatenate([mol_fp, mol_desc, route_f])


# ═══════════════════════════════════════════════════════
#  IMPROVEMENT 3: Multi-objective loss components
# ═══════════════════════════════════════════════════════

def ranking_loss(scores_pred: np.ndarray, scores_true: np.ndarray) -> float:
    """Pairwise ranking loss (hinge)."""
    n = len(scores_pred)
    loss = 0.0
    count = 0
    for i in range(n):
        for j in range(n):
            if scores_true[i] > scores_true[j]:
                margin = scores_pred[j] - scores_pred[i] + 1.0
                loss += max(0.0, margin)
                count += 1
    return loss / max(count, 1)


def synthesizability_loss(routes: List[SynthesisRoute],
                           scores_pred: np.ndarray) -> float:
    """Penalize high scores for synthetically inaccessible routes."""
    sa_scores = np.array([sa_score_approx(r.target_smiles) / 10 for r in routes])
    # Routes with SA > 0.7 (hard) should not get top scores
    penalty = np.mean(np.maximum(0, scores_pred - (1 - sa_scores)))
    return float(penalty)


def docking_consistency_loss(routes: List[SynthesisRoute],
                               scores_pred: np.ndarray) -> float:
    """Routes with better docking scores should rank higher."""
    dock_scores = np.array([abs(r.docking_score) for r in routes])
    if dock_scores.max() == 0:
        return 0.0
    dock_norm = dock_scores / dock_scores.max()
    # Pred rank should correlate with docking rank
    from scipy.stats import spearmanr
    try:
        corr, _ = spearmanr(scores_pred, dock_norm)
        return float(max(0, 1 - corr))
    except Exception:
        return 0.0


def md_stability_loss(routes: List[SynthesisRoute],
                       scores_pred: np.ndarray) -> float:
    """Stable MD trajectories (low RMSD) should favor higher scores."""
    rmsd_vals = np.array([r.md_rmsd for r in routes])
    if rmsd_vals.max() == 0:
        return 0.0
    rmsd_norm = rmsd_vals / (rmsd_vals.max() + 1e-9)
    # High RMSD = unstable = should not rank high
    penalty = np.mean(scores_pred * rmsd_norm)
    return float(penalty)


def multi_objective_loss(routes: List[SynthesisRoute],
                          scores_pred: np.ndarray,
                          scores_true: np.ndarray,
                          alpha: float = 0.3,
                          beta: float = 0.2,
                          gamma: float = 0.15) -> Dict[str, float]:
    """
    Improvement 3: Multi-objective combined loss.

    L = ranking_loss
      + α * synthesizability_loss
      + β * docking_consistency_loss
      + γ * md_stability_loss
    """
    L_rank  = ranking_loss(scores_pred, scores_true)
    L_synth = synthesizability_loss(routes, scores_pred)
    L_dock  = docking_consistency_loss(routes, scores_pred)
    L_md    = md_stability_loss(routes, scores_pred)
    L_total = L_rank + alpha * L_synth + beta * L_dock + gamma * L_md

    return {
        "total":                round(L_total, 5),
        "ranking_loss":         round(L_rank,  5),
        "synthesizability_loss":round(L_synth, 5),
        "docking_consistency":  round(L_dock,  5),
        "md_stability_loss":    round(L_md,    5),
        "alpha": alpha, "beta": beta, "gamma": gamma,
    }


# ═══════════════════════════════════════════════════════
#  BASELINE 1: XGBoost Reranker
# ═══════════════════════════════════════════════════════

class XGBoostReranker:
    """
    BASELINE: XGBoost route reranker.
    Treats reranking as regression on route quality score.

    Based on: XGBoost (Chen & Guestrin, KDD 2016)
    Input: Morgan FP (512 bits) + RDKit descriptors + route graph features
    """

    MODEL_NAME = "XGBoostReranker"

    def __init__(self, n_estimators: int = 200, max_depth: int = 6,
                 lr: float = 0.05):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=lr,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, routes: List[SynthesisRoute],
             labels: np.ndarray) -> Dict[str, float]:
        X = np.array([full_features(r) for r in routes])
        X = self.scaler.fit_transform(X)
        self.model.fit(X, labels)
        self._fitted = True
        preds = self.model.predict(X)
        return {
            "train_mse": float(np.mean((preds - labels) ** 2)),
            "model": self.MODEL_NAME,
        }

    def predict(self, routes: List[SynthesisRoute]) -> np.ndarray:
        if not self._fitted:
            self._auto_fit(routes)
        X = np.array([full_features(r) for r in routes])
        X = self.scaler.transform(X)
        return self.model.predict(X)

    def rank_routes(self, routes: List[SynthesisRoute]) -> List[SynthesisRoute]:
        scores = self.predict(routes)
        for r, s in zip(routes, scores):
            r.reranker_score = float(s)
        ranked = sorted(routes, key=lambda r: r.reranker_score, reverse=True)
        for i, r in enumerate(ranked):
            r.rank = i + 1
        return ranked

    def _auto_fit(self, routes: List[SynthesisRoute]):
        """Auto-generate labels from backend scores when no ground truth."""
        labels = np.array([r.backend_score for r in routes])
        self.fit(routes, labels)

    @property
    def feature_names(self) -> List[str]:
        return (
            [f"fp_{i}" for i in range(512)] +
            ["mw","logp","tpsa","hbd","hba","rings","rotbonds",
             "natoms","arom_rings","fsp3","sa_score","qed_proxy",
             "stereocenters","heavy_atoms"] +
            ["route_depth","step_count","avg_conf","min_conf",
             "purch_ratio","template_entropy","branch_factor",
             "rxn_diversity","commercial_bb","backend_score",
             "docking_abs","md_rmsd","md_stability"]
        )


# ═══════════════════════════════════════════════════════
#  BASELINE 2: MLP Reranker
# ═══════════════════════════════════════════════════════

class MLPReranker:
    """
    BASELINE: MLP-based route reranker.
    Uses same features as XGBoost but with neural network.
    """

    MODEL_NAME = "MLPReranker"

    def __init__(self, hidden_layers=(256, 128, 64), max_iter: int = 300):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, routes: List[SynthesisRoute],
             labels: np.ndarray) -> Dict[str, float]:
        X = np.array([full_features(r) for r in routes])
        X = self.scaler.fit_transform(X)
        # Convert to binary labels for classifier
        y = (labels > labels.mean()).astype(int)
        self.model.fit(X, y)
        self._fitted = True
        proba = self.model.predict_proba(X)[:, 1]
        return {"train_accuracy": float((proba > 0.5).mean() == y.mean()),
                "model": self.MODEL_NAME}

    def predict(self, routes: List[SynthesisRoute]) -> np.ndarray:
        if not self._fitted:
            self._auto_fit(routes)
        X = np.array([full_features(r) for r in routes])
        X = self.scaler.transform(X)
        return self.model.predict_proba(X)[:, 1]

    def rank_routes(self, routes: List[SynthesisRoute]) -> List[SynthesisRoute]:
        scores = self.predict(routes)
        for r, s in zip(routes, scores):
            r.reranker_score = float(s)
        ranked = sorted(routes, key=lambda r: r.reranker_score, reverse=True)
        for i, r in enumerate(ranked):
            r.rank = i + 1
        return ranked

    def _auto_fit(self, routes):
        labels = np.array([r.backend_score for r in routes])
        self.fit(routes, labels)


# ═══════════════════════════════════════════════════════
#  OUR MODEL: Dual-Branch Route Ranker
# ═══════════════════════════════════════════════════════

class DualBranchRouteRanker:
    """
    OUR IMPROVED MODEL: Dual-Branch Route Prioritization Network.

    Architecture:
    ┌────────────────────────────────────────────────────┐
    │  Input: (target SMILES, synthesis route)           │
    │                                                    │
    │  Branch 1 (Molecule Branch):                       │
    │    Morgan FP (512) + RDKit Descriptors (14)        │
    │    → Dense(256) → ReLU → Dense(128)                │
    │                                                    │
    │  Branch 2 (Route Branch):                          │
    │    Route graph features (13)                       │
    │    → Dense(64) → ReLU → Dense(64)                  │
    │                                                    │
    │  Fusion: Gated cross-attention                     │
    │    gate = σ(W_g · [mol_emb; route_emb])            │
    │    fused = gate * mol_emb + (1-gate) * route_emb   │
    │    → Dense(64) → Dense(1) → score                  │
    └────────────────────────────────────────────────────┘

    Training: multi-objective loss (Improvement 3)
    Features: Improvement 4 route graph features
    """

    MODEL_NAME = "DualBranchRouteRanker"

    def __init__(self, mol_dim: int = 526, route_dim: int = 13,
                 hidden_mol: int = 128, hidden_route: int = 64,
                 fusion_dim: int = 64, lr: float = 0.001,
                 n_epochs: int = 50, alpha: float = 0.3,
                 beta: float = 0.2, gamma: float = 0.15):
        np.random.seed(42)
        self.mol_dim = mol_dim       # 512 FP + 14 desc
        self.route_dim = route_dim
        self.hidden_mol = hidden_mol
        self.hidden_route = hidden_route
        self.fusion_dim = fusion_dim
        self.lr = lr
        self.n_epochs = n_epochs
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # ── Molecule branch weights ──
        self.W_mol1 = np.random.randn(mol_dim, hidden_mol) * np.sqrt(2 / mol_dim)
        self.b_mol1 = np.zeros(hidden_mol)
        self.W_mol2 = np.random.randn(hidden_mol, hidden_mol) * np.sqrt(2 / hidden_mol)
        self.b_mol2 = np.zeros(hidden_mol)

        # ── Route branch weights ──
        self.W_rte1 = np.random.randn(route_dim, hidden_route) * np.sqrt(2 / route_dim)
        self.b_rte1 = np.zeros(hidden_route)
        self.W_rte2 = np.random.randn(hidden_route, hidden_route) * np.sqrt(2 / hidden_route)
        self.b_rte2 = np.zeros(hidden_route)

        # ── Gated fusion ──
        gate_in = hidden_mol + hidden_route
        self.W_gate = np.random.randn(gate_in, hidden_mol) * 0.01
        self.b_gate = np.zeros(hidden_mol)

        # ── Output ──
        self.W_out1 = np.random.randn(hidden_mol, fusion_dim) * np.sqrt(2 / hidden_mol)
        self.b_out1 = np.zeros(fusion_dim)
        self.W_out2 = np.random.randn(fusion_dim, 1) * np.sqrt(2 / fusion_dim)
        self.b_out2 = np.zeros(1)

        self.scaler_mol   = StandardScaler()
        self.scaler_route = StandardScaler()
        self._fitted = False
        self.loss_history: List[Dict] = []

    @staticmethod
    def _relu(x):  return np.maximum(0, x)
    @staticmethod
    def _sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

    def _forward_mol(self, X_mol: np.ndarray) -> np.ndarray:
        h = self._relu(X_mol @ self.W_mol1 + self.b_mol1)
        return self._relu(h @ self.W_mol2 + self.b_mol2)

    def _forward_route(self, X_rte: np.ndarray) -> np.ndarray:
        h = self._relu(X_rte @ self.W_rte1 + self.b_rte1)
        return self._relu(h @ self.W_rte2 + self.b_rte2)

    def _fuse(self, mol_emb: np.ndarray, rte_emb: np.ndarray) -> np.ndarray:
        """Gated fusion: gate controls how much of each branch contributes."""
        concat = np.concatenate([mol_emb, rte_emb], axis=-1)
        gate   = self._sigmoid(concat @ self.W_gate + self.b_gate)
        # Pad/trim route_emb to match mol_emb dim
        if rte_emb.shape[-1] < mol_emb.shape[-1]:
            pad = mol_emb.shape[-1] - rte_emb.shape[-1]
            rte_padded = np.pad(rte_emb, [(0,0),(0,pad)] if rte_emb.ndim==2 else [(0,pad)])
        else:
            rte_padded = rte_emb[..., :mol_emb.shape[-1]]
        return gate * mol_emb + (1 - gate) * rte_padded

    def _forward(self, X_mol: np.ndarray, X_rte: np.ndarray) -> np.ndarray:
        mol_emb = self._forward_mol(X_mol)           # (N, hidden_mol)
        rte_emb = self._forward_route(X_rte)         # (N, hidden_route)
        fused   = self._fuse(mol_emb, rte_emb)       # (N, hidden_mol)
        h_out   = self._relu(fused @ self.W_out1 + self.b_out1)
        scores  = self._sigmoid(h_out @ self.W_out2 + self.b_out2)
        return scores.squeeze(-1)

    def _prepare_features(self, routes: List[SynthesisRoute],
                            fit_scaler: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        X_mol   = np.array([np.concatenate([morgan_fp(r.target_smiles, 512),
                                             rdkit_desc(r.target_smiles)])
                             for r in routes])
        X_route = np.array([route_tabular_features(r) for r in routes])
        if fit_scaler:
            X_mol   = self.scaler_mol.fit_transform(X_mol)
            X_route = self.scaler_route.fit_transform(X_route)
        else:
            X_mol   = self.scaler_mol.transform(X_mol)
            X_route = self.scaler_route.transform(X_route)
        return X_mol, X_route

    def fit(self, routes: List[SynthesisRoute],
             labels: np.ndarray) -> Dict:
        """Train with gradient-free evolutionary + gradient approximation."""
        X_mol, X_rte = self._prepare_features(routes, fit_scaler=True)
        self._fitted = True

        best_loss = float("inf")
        best_weights = self._get_weights()

        for epoch in range(self.n_epochs):
            preds = self._forward(X_mol, X_rte)
            loss_dict = multi_objective_loss(
                routes, preds, labels,
                alpha=self.alpha, beta=self.beta, gamma=self.gamma
            )
            total = loss_dict["total"]
            self.loss_history.append(loss_dict)

            if total < best_loss:
                best_loss = total
                best_weights = self._get_weights()

            # Gradient-free weight update (evolution strategy)
            noise_scale = self.lr * max(0.1, 1 - epoch / self.n_epochs)
            self._perturb_weights(noise_scale, preds, labels)

            if (epoch + 1) % 10 == 0:
                print(f"    [DualBranch] Epoch {epoch+1:>3}/{self.n_epochs} "
                      f"| Loss={total:.4f} "
                      f"(rank={loss_dict['ranking_loss']:.4f}, "
                      f"synth={loss_dict['synthesizability_loss']:.4f})")

        self._set_weights(best_weights)
        final_preds = self._forward(X_mol, X_rte)
        return {
            "model": self.MODEL_NAME,
            "final_loss": best_loss,
            "loss_history": self.loss_history,
            "final_predictions": final_preds.tolist(),
        }

    def _get_weights(self):
        return [w.copy() for w in [
            self.W_mol1, self.b_mol1, self.W_mol2, self.b_mol2,
            self.W_rte1, self.b_rte1, self.W_rte2, self.b_rte2,
            self.W_gate, self.b_gate,
            self.W_out1, self.b_out1, self.W_out2, self.b_out2,
        ]]

    def _set_weights(self, weights):
        (self.W_mol1, self.b_mol1, self.W_mol2, self.b_mol2,
         self.W_rte1, self.b_rte1, self.W_rte2, self.b_rte2,
         self.W_gate, self.b_gate,
         self.W_out1, self.b_out1, self.W_out2, self.b_out2) = weights

    def _perturb_weights(self, scale: float, preds: np.ndarray,
                          labels: np.ndarray):
        """Simple gradient approximation via noise perturbation."""
        for attr in ["W_mol1","W_mol2","W_rte1","W_rte2",
                     "W_gate","W_out1","W_out2"]:
            w = getattr(self, attr)
            setattr(self, attr, w - scale * np.sign(preds.mean() - labels.mean())
                    * np.random.randn(*w.shape) * 0.01)

    def predict(self, routes: List[SynthesisRoute]) -> np.ndarray:
        if not self._fitted:
            self._auto_fit(routes)
        X_mol, X_rte = self._prepare_features(routes, fit_scaler=False)
        return self._forward(X_mol, X_rte)

    def rank_routes(self, routes: List[SynthesisRoute]) -> List[SynthesisRoute]:
        scores = self.predict(routes)
        for r, s in zip(routes, scores):
            r.reranker_score = float(s)
        ranked = sorted(routes, key=lambda r: r.reranker_score, reverse=True)
        for i, r in enumerate(ranked):
            r.rank = i + 1
        return ranked

    def _auto_fit(self, routes):
        labels = np.array([r.backend_score for r in routes])
        self.fit(routes, labels)
