# RetroDock-X

**A modular platform for retrosynthesis planning and route prioritization guided by docking and molecular dynamics**

**一个基于对接与分子动力学引导的逆合成规划与路径优先级排序模块化平台**

> Dual-backend retrosynthesis (ASKCOS + AiZynthFinder) × ML reranking (XGBoost / MLP / DualBranch) × SHAP explainability × macaroon-palette publication figures

RetroDock-X 是一个集成化的计算化学平台，融合了**双后端逆合成规划（ASKCOS + AiZynthFinder）**、**机器学习路径重排序**以及**SHAP 可解释性分析**。该系统能够结合分子对接和分子动力学信息，对合成路径进行多目标优化与优先级排序，并生成高质量可发表图表（macaroon 配色风格）。

### 说的人话：就是说：

RetroDock-X 是一个面向 AI 药物设计（AIDD）的模块化计算平台，旨在实现从目标分子输入到最优合成路径筛选的端到端自动化流程。该系统集成了基于模板的逆合成规划算法（ASKCOS 与 AiZynthFinder）、分子对接（molecular docking）以及分子动力学（molecular dynamics, MD）信息，并通过机器学习模型对候选合成路径进行多目标优化排序。

在逆合成阶段，平台通过统一的抽象接口封装 ASKCOS 与 AiZynthFinder，实现不同规划后端之间的无缝切换与批量调用。系统为每个目标分子生成多条候选合成路径，并构建对应的合成树结构（synthesis tree）。

在路径评估与排序阶段，RetroDock-X 提出了基于多模态特征融合的机器学习重排序框架。该框架综合利用分子级特征（如 ECFP 指纹与 RDKit 分子描述符）与路径级图特征（包括路径深度、反应步骤数、模板置信度、可购前体比例及反应类别多样性等共 13 维特征），并引入对接评分与 MD 稳定性指标，从而实现对“合成可行性”与“生物活性潜力”的联合建模。

系统实现了三类路径重排序模型：基于梯度提升的 XGBoost 模型、前馈神经网络（MLP）模型，以及本文提出的 DualBranchRouteRanker。DualBranch 模型采用双分支结构分别编码分子特征与路径特征，并通过门控融合机制（gated fusion / cross-attention）实现跨模态信息交互，从而提升路径评分的判别能力。

此外，RetroDock-X 设计了多目标损失函数，将排序损失与合成可行性损失、对接一致性损失以及 MD 稳定性损失进行加权联合优化，实现多目标约束下的路径优选。

为增强模型的可解释性，系统引入 SHAP（SHapley Additive exPlanations）框架，对不同模型进行特征贡献分析，支持全局特征重要性评估与单路径决策解释，从而提高模型在计算化学与药物设计场景中的可信度与可用性。

实验结果表明，在 USPTO-50k 子集上，DualBranch 模型在 Top-k 准确率、路径有效性以及可购前体比例等指标上均显著优于传统基线方法，验证了多模态融合与多目标优化策略的有效性。

综上，RetroDock-X 提供了一个融合逆合成规划、结构生物学信息与机器学习决策的统一框架，为计算驱动的合成路径设计与药物开发提供了系统性解决方案。



### Project Description (English)：

RetroDock-X is a modular computational platform for AI-driven drug discovery (AIDD), designed to provide an end-to-end pipeline from target molecule input to optimal synthetic route prioritization. The system integrates template-based retrosynthesis planning engines (ASKCOS and AiZynthFinder), molecular docking, and molecular dynamics (MD) simulations, and employs machine learning models for multi-objective route reranking.

At the retrosynthesis stage, RetroDock-X introduces a unified abstraction layer to standardize interactions with ASKCOS and AiZynthFinder, enabling seamless backend switching and scalable batch processing. For each target molecule, multiple candidate synthetic routes are generated and represented as synthesis trees.

For route evaluation and prioritization, the platform implements a multi-modal machine learning framework that jointly models molecular-level and route-level information. Molecular features include ECFP fingerprints and RDKit descriptors, while route-level graph features consist of 13 engineered attributes such as route depth, step count, template confidence, purchasable precursor ratio, and reaction class diversity. Additionally, docking scores and MD-derived stability metrics are incorporated to capture biological relevance.

Three reranking models are implemented: a gradient boosting model (XGBoost), a feedforward neural network (MLP), and a novel DualBranchRouteRanker. The proposed DualBranch architecture employs two parallel branches to encode molecular and route features, respectively, followed by a gated fusion mechanism (cross-attention) to enable effective multi-modal interaction and improve predictive performance.

A multi-objective loss function is introduced to jointly optimize ranking performance, synthesizability, docking consistency, and MD stability, enabling balanced optimization across chemical feasibility and biological activity.

To enhance interpretability, the platform integrates SHAP (SHapley Additive exPlanations) for both global and local explanation of model predictions, providing insights into feature contributions at both dataset and individual route levels.

Experimental results on the USPTO-50k subset demonstrate that the proposed DualBranch model significantly outperforms baseline approaches in Top-k accuracy, route validity, and purchasable precursor ratio, highlighting the effectiveness of multi-modal feature fusion and multi-objective optimization.

Overall, RetroDock-X establishes a unified framework that bridges retrosynthesis planning, structural bioinformatics, and machine learning, offering a systematic solution for computationally guided synthesis design and drug development.



### 其实一句话：

构建了一个融合逆合成规划（ASKCOS/AiZynthFinder）、分子对接与分子动力学的 AI 药物设计平台，通过多模态特征与多目标学习实现合成路径优先级排序，并基于 SHAP 提供模型可解释性分析。



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
