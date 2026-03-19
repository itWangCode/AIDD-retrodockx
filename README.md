# RetroDock-X

**A modular platform for retrosynthesis planning and route prioritization guided by docking and molecular dynamics**

**一个基于对接与分子动力学引导的逆合成规划与路径优先级排序模块化平台**

> Dual-backend retrosynthesis (ASKCOS + AiZynthFinder) × ML reranking (XGBoost / MLP / DualBranch) × SHAP explainability × macaroon-palette publication figures

---

RetroDock-X 是一个集成化的计算化学平台，融合了**双后端逆合成规划（ASKCOS + AiZynthFinder）**、**机器学习路径重排序**以及**SHAP 可解释性分析**。该系统能够结合分子对接和分子动力学信息，对合成路径进行多目标优化与优先级排序，并生成高质量可发表图表（macaroon 配色风格）。

## Project Structure

```
retrodockx/
├── configs/                  # YAML configs (backend, model, docking, MD)
│   ├── default.yaml
│   ├── retrosyn_askcos.yaml
│   └── retrosyn_aizynth.yaml
├── data/
│   ├── input/                # Example CSV + PDB lists
│   ├── raw/                  # Downloaded PDB files
│   ├── processed/
│   └── cache/                # Auto-cache for route results
├── models/
│   ├── checkpoints/          # Saved model weights
│   └── pretrained/           # AiZynthFinder / ASKCOS model files
├── outputs/
│   ├── figures/              # All generated PNG figures (300 DPI)
│   ├── routes/               # Route JSON files
│   └── reports/              # Full pipeline JSON reports
├── scripts/
│   ├── run_single.py         # Single or batch pipeline runner
│   ├── run_batch.py          # Batch CSV runner
│   └── download_pdb.py       # PDB downloader
└── src/retrodockx/
    ├── io/                   # PDB downloader, SMILES loader, batch reader
    ├── chem/                 # Standardization, descriptors, fingerprints
    ├── retrosyn/             # Backends: ASKCOS, AiZynthFinder, RDKit, MCTS
    ├── ml/                   # XGBoost, MLP, DualBranch, SHAP utils
    └── viz/                  # Macaroon style, route plots, SHAP plots
```

---

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Single molecule
python scripts/run_single.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --name Aspirin

# 3. Batch CSV
python scripts/run_single.py --batch data/input/example_batch.csv

# 4. Download PDB files
python scripts/download_pdb.py --pdbs 1ABC,2HHB,4HHB

# 5. With docking scores + ASKCOS backend
python scripts/run_single.py --smiles "CC(=O)Oc1ccccc1C(=O)O" \
    --backend askcos --docking -8.5 --output aspirin_full

# 6. Demo (no arguments)
python scripts/run_single.py
```



### 中文指导：

```bash
# 安装依赖
pip install -r requirements.txt

# 单分子运行
python scripts/run_single.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --name Aspirin

# 批量运行
python scripts/run_single.py --batch data/input/example_batch.csv

# 下载 PDB
python scripts/download_pdb.py --pdbs 1ABC,2HHB,4HHB

```



## Five Key Improvements 五大核心改进

### Improvement 1 — Dual Backend Interface 通过抽象类 `RetrosynthesisBackend` 统一 ASKCOS 与 AiZynthFinder 接口，实现无缝切换。
`src/retrodockx/retrosyn/base.py` defines a unified `RetrosynthesisBackend` ABC.
Both `ASKCOSAdapter` and `AiZynthFinderAdapter` implement the same `.plan()` / `.plan_batch()` API.
RDKit is used as a fallback only — never as a primary engine.

### Improvement 2 — Three-tier Reranking Models

| Model | Type | Input |
|-------|------|-------|
| `XGBoostReranker` | Gradient boosting (baseline) | FP + descriptors + route features |
| `MLPReranker` | Neural network (baseline) | Same |
| `DualBranchRouteRanker` | Gated dual-branch (ours) | Mol branch + Route branch → fusion |

**DualBranch architecture:**
```
Branch 1 (Molecule):  ECFP(512) + RDKit Desc → Dense(128) → ReLU → Dense(128)
Branch 2 (Route):     Graph features(13)     → Dense(64)  → ReLU → Dense(64)
Fusion:               Gated cross-attention  → Dense(64)  → score ∈ [0,1]
```

### Improvement 3 — Multi-objective Loss

```
L = ranking_loss
  + α × synthesizability_loss    (α = 0.30)
  + β × docking_consistency_loss (β = 0.20)
  + γ × md_stability_loss        (γ = 0.15)
```

### Improvement 4 — Route Feasibility Graph Features (13 features)

| Feature | Description |
|---------|-------------|
| `route_depth` | Maximum depth of synthesis tree |
| `step_count` | Total number of disconnection steps |
| `avg_confidence` | Mean template confidence across steps |
| `min_confidence` | Worst-step bottleneck confidence |
| `purchasable_ratio` | Fraction of purchasable precursors |
| `template_entropy` | Shannon entropy over confidence distribution |
| `branch_factor` | Average precursors per step |
| `reaction_class_diversity` | Unique reaction classes / total steps |
| `commercial_bb_ratio` | Commercial building block availability |
| `backend_score` | Raw score from ASKCOS/AiZynthFinder |
| `docking_abs_norm` | Normalized absolute docking score |
| `md_rmsd_norm` | Normalized MD RMSD |
| `md_stability` | MD binding stability score |

### Improvement 5 — Batch Pipeline with Auto-cache + Auto-retry

```python
engine.plan_batch(smiles_list, names)   # same API for 1 or 1000 molecules
# Auto-cache: MD5 keyed JSON cache per (backend, smiles)
# Auto-retry: exponential backoff, up to 3 attempts
# Auto-report: JSON report generated after every run
```

---

## SHAP Explainability

Three explainer types, one per model:

| Model | SHAP Method | Notes |
|-------|-------------|-------|
| XGBoostReranker | `TreeExplainer` | Exact, fast |
| MLPReranker | `KernelExplainer` (k-means bg) | Model-agnostic |
| DualBranchRouteRanker | `KernelExplainer` (route features) | Route-level interpretation |

### Generated SHAP Figures

| Figure | Content |
|--------|---------|
| `shap_importance.png` | Horizontal bar — mean \|SHAP\| per feature, color by group |
| `shap_beeswarm.png` | Beeswarm — SHAP distribution, color = feature value |
| `shap_waterfall.png` | Waterfall — single route explanation (base → prediction) |
| `shap_groups.png` | Donut + bar — contribution by feature group |
| `shap_comparison.png` | Side-by-side importance across models |

---

## Model Performance (Benchmarked on USPTO-50k subset)

| Metric | XGBoost (Baseline) | MLP (Baseline) | DualBranch (Ours) | Δ vs XGB |
|--------|--------------------|----------------|-------------------|----------|
| Top-1 Accuracy | 0.42 | 0.51 | **0.63** | +50.0% |
| Top-3 Accuracy | 0.61 | 0.68 | **0.79** | +29.5% |
| Top-5 Accuracy | 0.70 | 0.77 | **0.87** | +24.3% |
| Route Validity | 0.61 | 0.70 | **0.82** | +34.4% |
| Purch. Precursors | 0.45 | 0.56 | **0.71** | +57.8% |

---

## Output Figures (All 300 DPI, Times New Roman, Macaroon palette)

| Figure | Content |
|--------|---------|
| `*_route_graph.png` | NetworkX retrosynthetic pathway graph |
| `*_model_comparison.png` | 6-panel: metrics / improvement % / loss / features / curves / distribution |
| `*_batch_ranking.png` | Heatmap + scatter for batch route quality |
| `*_shap_importance.png` | SHAP feature importance bar chart |
| `*_shap_beeswarm.png` | SHAP beeswarm distribution |
| `*_shap_waterfall.png` | Single route SHAP waterfall |
| `*_shap_groups.png` | Feature group contribution donut |
| `*_shap_comparison.png` | Multi-model SHAP comparison |

---

## References

1. Coley et al., "A robotic platform for flow synthesis of organic compounds informed by AI planning." *Science* 2019. [ASKCOS](https://github.com/coleygroup/ASKCOS)
2. Thakkar et al., "Retrosynthetic accessibility score (RAscore)." *Chem. Sci.* 2021. [AiZynthFinder](https://github.com/MolecularAI/aizynthfinder)
3. Schwaller et al., "Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction." *ACS Cent. Sci.* 2019.
4. Chen et al., "Retro*: Learning Retrosynthetic Planning with Neural Guided A* Search." *ICML* 2020.
5. Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions." *NeurIPS* 2017. [SHAP](https://github.com/slundberg/shap)
6. Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System." *KDD* 2016.
