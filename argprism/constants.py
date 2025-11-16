"""Shared constants for the ARGprism package."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / "data"
MODELS_DIR = PACKAGE_ROOT / "models"

PROT_ALBERT_MODEL = "Rostlab/prot_albert"

DEFAULT_CLASSIFIER_PATH = MODELS_DIR / "best_model_fold4.pth"
DEFAULT_ARG_DB = DATA_DIR / "ARGPrismDB.fasta"
DEFAULT_METADATA = DATA_DIR / "metadata_arg.json"
DEFAULT_OUTPUT_FASTA = "predicted_ARGs.fasta"
DEFAULT_DIAMOND_PREFIX = "diamond_arg_db"
DEFAULT_DIAMOND_OUTPUT = "predicted_ARGs_vs_ref.tsv"
DEFAULT_REPORT = "final_ARG_prediction_report.csv"