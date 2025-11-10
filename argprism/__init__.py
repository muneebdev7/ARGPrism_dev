"""
ARGprism: Deep Learning-based Antibiotic Resistance Gene Prediction Pipeline
"""

__version__ = "1.0.0"
__author__ = "Muneeb"

from .pipeline import ARGPrismPipeline
from .classifier import ARGClassifier
from .embeddings import generate_embedding, generate_embeddings

__all__ = [
    'ARGPrismPipeline',
    'ARGClassifier',
    'generate_embedding',
    'generate_embeddings',
]
