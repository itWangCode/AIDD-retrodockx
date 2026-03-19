"""Basic smoke tests for RetroDock-X pipeline."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

def test_smiles_validation():
    from retrodockx.chem.standardize import is_valid, standardize
    assert is_valid("CC(=O)Oc1ccccc1C(=O)O")
    assert not is_valid("not_a_smiles")
    assert standardize("C(=O)(O)c1ccccc1OC(C)=O") != ""

def test_rdkit_route():
    from retrodockx.retrosyn.rdkit_route_utils import RDKitRouteUtils
    ru = RDKitRouteUtils()
    routes = ru.run_retrosynthesis("CC(=O)Oc1ccccc1C(=O)O", max_routes=3)
    assert isinstance(routes, list)

def test_feature_extraction():
    from retrodockx.retrosyn.base import SynthesisRoute, ReactionStep
    step = ReactionStep(smiles="CC(=O)Oc1ccccc1C(=O)O",
                        precursors=["CC(=O)O", "Oc1ccccc1C(=O)O"],
                        reaction_class="Hydrolysis", confidence=0.90,
                        depth=0, source="test")
    route = SynthesisRoute(target_smiles="CC(=O)Oc1ccccc1C(=O)O",
                            target_name="Aspirin", steps=[step],
                            backend="test", backend_score=0.9)
    fv = route.feature_vector()
    assert "avg_confidence" in fv
    assert fv["step_count"] == 1

def test_xgboost_reranker():
    import numpy as np
    from retrodockx.retrosyn.base import SynthesisRoute, ReactionStep
    from retrodockx.ml.baselines import XGBoostReranker
    step = ReactionStep(smiles="CC(=O)Oc1ccccc1C(=O)O",
                        precursors=["CC(=O)O"], confidence=0.8, depth=0, source="t")
    routes = [SynthesisRoute(target_smiles="CC(=O)Oc1ccccc1C(=O)O",
                              target_name=f"R{i}", steps=[step],
                              backend="t", backend_score=0.8 - i*0.1)
               for i in range(6)]
    labels = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    model = XGBoostReranker(n_estimators=20)
    res = model.fit(routes, labels)
    assert "train_mse" in res
    ranked = model.rank_routes(routes)
    assert ranked[0].rank == 1

if __name__ == "__main__":
    test_smiles_validation(); print("✓ smiles_validation")
    test_rdkit_route();       print("✓ rdkit_route")
    test_feature_extraction();print("✓ feature_extraction")
    test_xgboost_reranker();  print("✓ xgboost_reranker")
    print("\nAll tests passed!")
