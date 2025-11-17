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
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return self.softmax(x)


def load_classifier(path: Path, device: torch.device) -> ARGClassifier:
    classifier = ARGClassifier()
    state_dict = torch.load(path, map_location=device)
    classifier.load_state_dict(state_dict)
    classifier.to(device)
    # Set to train mode to use batch statistics rather than stored running stats
    # This appears to fix the prediction bias issue
    classifier.train()
    return classifier


def classify_embeddings(
    embeddings: Dict[str, torch.Tensor],
    classifier: ARGClassifier,
    device: torch.device,
) -> Dict[str, str]:
    results: Dict[str, str] = {}
    
    # Collect all embeddings first for batch processing
    seq_ids = list(embeddings.keys())
    all_embeddings = torch.stack([embeddings[seq_id].to(device=device, dtype=torch.float32) for seq_id in seq_ids])
    
    with torch.no_grad():
        # Process in batches to avoid BatchNorm issues with single samples
        batch_size = 32
        for i in range(0, len(all_embeddings), batch_size):
            batch_embeddings = all_embeddings[i:i+batch_size]
            batch_outputs = classifier(batch_embeddings)
            batch_preds = torch.argmax(batch_outputs, dim=1)
            
            for j, pred_class in enumerate(batch_preds):
                seq_id = seq_ids[i + j]
                results[seq_id] = "ARG" if pred_class.item() == 1 else "Non-ARG"
                
    return results
