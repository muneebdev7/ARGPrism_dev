"""Shared constants for the ARGprism package."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PACKAGE_ROOT / "data"
MODELS_DIR = PACKAGE_ROOT / "models"

PRETRAINED_PROTEIN_LANGUAGE_MODEL = "Rostlab/prot_albert"

DEFAULT_ARG_CLASSIFIER_MODEL_PATH = MODELS_DIR / "best_model_fold4.pth"
DEFAULT_ARG_REFERENCE_DATABASE_FASTA = DATA_DIR / "ARGPrismDB.fasta"
DEFAULT_ARG_METADATA_JSON = DATA_DIR / "metadata_arg.json"
DEFAULT_PREDICTED_ARGS_FASTA_FILENAME = "predicted_query_ARGs.fasta"
DEFAULT_DIAMOND_DATABASE_PREFIX = "diamond_arg_db"
DEFAULT_DIAMOND_BLAST_OUTPUT_FILENAME = "predicted_query_ARGs_vs_reference_ARGs.tsv"
DEFAULT_FINAL_REPORT_CSV_FILENAME = "final_ARGs_prediction_report.csv"