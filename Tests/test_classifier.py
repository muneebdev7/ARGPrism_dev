"""Basic classifier loading regression tests."""

from pathlib import Path

import torch

from argprism.classifier import ARGClassifier, load_classifier

MODEL_PATH = Path(__file__).resolve().parents[1] / "argprism" / "models" / "best_model_fold4.pth"


def test_classifier_architecture_matches_checkpoint() -> None:
    classifier = ARGClassifier()
    state = torch.load(MODEL_PATH, map_location="cpu")
    missing = set(classifier.state_dict().keys()) - set(state.keys())
    unexpected = set(state.keys()) - set(classifier.state_dict().keys())
    assert not missing, f"missing keys: {sorted(missing)}"
    assert not unexpected, f"unexpected keys: {sorted(unexpected)}"


def test_classifier_forward_pass() -> None:
    device = torch.device("cpu")
    classifier = load_classifier(MODEL_PATH, device)
    dummy = torch.randn(1, 4096, device=device)
    output = classifier(dummy)
    assert output.shape == (1, 2)
    assert torch.allclose(output.sum(dim=1), torch.ones(1, device=device), atol=1e-5)
