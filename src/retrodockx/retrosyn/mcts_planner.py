"""
src/retrodockx/retrosyn/mcts_planner.py
========================================
Monte Carlo Tree Search retrosynthesis planner.
Used as AiZynthFinder fallback.
Improved from: Chen et al. Retro* (2020)
"""

import math
import random
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field
from rdkit import Chem
from rdkit.Chem import Descriptors

from retrodockx.retrosyn.base import SynthesisRoute, ReactionStep


@dataclass
class MCTSNode:
    smiles: str
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    reaction: str = ""
    reaction_class: str = ""
    confidence: float = 0.0
    depth: int = 0
    is_terminal: bool = False

    def ucb1(self, c: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else 1
        return (self.value / self.visits
                + c * math.sqrt(math.log(parent_visits) / self.visits))

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class MCTSPlanner:
    """MCTS retrosynthesis with UCB1 tree policy."""

    def __init__(self, n_simulations: int = 60, max_depth: int = 5,
                 rdkit_utils=None):
        self.n_sims = n_simulations
        self.max_depth = max_depth
        self.ru = rdkit_utils

    def search(self, smiles: str, n_routes: int = 5) -> List[SynthesisRoute]:
        root = MCTSNode(smiles=smiles, depth=0)

        for _ in range(self.n_sims):
            node = self._select(root)
            if not node.is_terminal:
                node = self._expand(node)
            value = self._simulate(node)
            self._backprop(node, value)

        return self._extract_routes(root, smiles, n_routes)

    def _select(self, node: MCTSNode) -> MCTSNode:
        while not node.is_leaf() and not node.is_terminal:
            node = max(node.children, key=lambda n: n.ucb1())
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        if node.depth >= self.max_depth:
            node.is_terminal = True
            return node
        steps = self.ru.analyze_one(node.smiles, node.depth) if self.ru else []
        for step in steps[:3]:
            for prec in step.precursors[:2]:
                mol = Chem.MolFromSmiles(prec)
                terminal = (
                    mol is None or
                    node.depth + 1 >= self.max_depth or
                    (mol is not None and mol.GetNumHeavyAtoms() <= 6)
                )
                child = MCTSNode(
                    smiles=prec, parent=node,
                    reaction=step.template_id,
                    reaction_class=step.reaction_class,
                    confidence=step.confidence,
                    depth=node.depth + 1,
                    is_terminal=terminal,
                )
                node.children.append(child)
        if not node.children:
            node.is_terminal = True
        return random.choice(node.children) if node.children else node

    def _simulate(self, node: MCTSNode) -> float:
        if node.is_terminal:
            mol = Chem.MolFromSmiles(node.smiles)
            return 0.95 if mol and mol.GetNumHeavyAtoms() <= 8 else 0.3
        cur = node.smiles
        score = 0.5
        for _ in range(self.max_depth - node.depth):
            mol = Chem.MolFromSmiles(cur)
            if mol is None:
                break
            if mol.GetNumHeavyAtoms() <= 6:
                score = 0.95
                break
            steps = self.ru.analyze_one(cur) if self.ru else []
            if not steps:
                score = 0.2
                break
            best = random.choice(steps[:min(3, len(steps))])
            score = best.confidence
            cur = best.precursors[0] if best.precursors else cur
        return score

    def _backprop(self, node: MCTSNode, value: float):
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    def _extract_routes(self, root: MCTSNode, smiles: str,
                          n_routes: int) -> List[SynthesisRoute]:
        routes = []
        self._dfs_routes(root, [], routes, smiles, n_routes)
        routes.sort(key=lambda r: r.backend_score, reverse=True)
        return routes[:n_routes]

    def _dfs_routes(self, node: MCTSNode, path: list,
                     routes: List[SynthesisRoute], smiles: str, n_routes: int):
        if len(routes) >= n_routes:
            return
        if node.is_terminal or not node.children:
            if path:
                route = SynthesisRoute(
                    target_smiles=smiles,
                    target_name=f"MCTS_Route_{len(routes)+1}",
                    steps=list(path),
                    backend="mcts_rdkit",
                    backend_score=node.value / max(node.visits, 1),
                )
                route.compute_graph_features()
                routes.append(route)
            return
        for child in sorted(node.children,
                             key=lambda n: n.value / max(n.visits, 1),
                             reverse=True)[:2]:
            step = ReactionStep(
                smiles=node.smiles,
                precursors=[child.smiles],
                template_id=child.reaction,
                reaction_class=child.reaction_class,
                confidence=child.confidence,
                depth=child.depth - 1,
                source="mcts",
            )
            self._dfs_routes(child, path + [step], routes, smiles, n_routes)
