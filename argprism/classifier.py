"""Classifier utilities for ARGprism."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch import nn


class ARGClassifier(nn.Module):
    """Simple feed-forward classifier trained on ProtAlbert embeddings."""

    def __init__(self, input_dim: int = 4096) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(128, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.out(x)
        return self.softmax(x)


def load_classifier(path: Path, device: torch.device) -> ARGClassifier:
    classifier = ARGClassifier()
    state_dict = torch.load(path, map_location=device)
    classifier.load_state_dict(state_dict)
    classifier.to(device)
    classifier.eval()
    return classifier


def classify_embeddings(
    embeddings: Dict[str, torch.Tensor],
    classifier: ARGClassifier,
    device: torch.device,
) -> Dict[str, str]:
    results: Dict[str, str] = {}
    with torch.no_grad():
        for seq_id, embedding in embeddings.items():
            tensor = embedding.to(device=device, dtype=torch.float32)
            output = classifier(tensor.unsqueeze(0))
            pred_class = torch.argmax(output, dim=1).item()
            results[seq_id] = "ARG" if pred_class == 1 else "Non-ARG"
    return results
