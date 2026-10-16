import os

# Paths
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SPIKE_DIR  = BASE_DIR
THEMES_DIR = os.path.join(SPIKE_DIR, "themes")

# docling models — pre-downloaded via download_docling_models.py
# Used as pipeline_options.artifacts_path so docling never hits HuggingFace at runtime.
DOCLING_MODELS_PATH = os.path.expanduser(
    os.getenv("DOCLING_MODELS_PATH", os.path.join(SPIKE_DIR, ".docling-models"))
)

# AWS Bedrock — Llama as chat model
BEDROCK_REGION = ""
CHAT_MODEL     = ""