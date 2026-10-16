"""
download_docling_models.py — Pre-download docling models to a local directory.

Run this ONCE on a network that allows huggingface.co (e.g. phone hotspot or
home WiFi). Afterwards, read_themes.py reads the models from this directory via
config.DOCLING_MODELS_PATH and never reaches out to HuggingFace at runtime.

Usage:
    python download_docling_models.py
    python download_docling_models.py --models-path /custom/dir

Only the models actually used by read_themes.py are downloaded:
  • layout            — page layout detection
  • tableformer       — table structure recognition
  • picture_classifier — chart/graph/map classification
  • easyocr           — OCR backend

Heavy/optional models are skipped (code_formula, smolvlm, smoldocling,
granite_vision) to keep the download under ~500 MB.
"""

import truststore
truststore.inject_into_ssl()

import argparse
import sys
from pathlib import Path

from docling.utils.model_downloader import download_models

import config


def main():
    parser = argparse.ArgumentParser(description="Pre-download docling models for offline use.")
    parser.add_argument(
        "--models-path",
        type=Path,
        default=Path(config.DOCLING_MODELS_PATH),
        help=f"Where to save the models (default: {config.DOCLING_MODELS_PATH})",
    )
    args = parser.parse_args()

    models_path = args.models_path.expanduser().resolve()
    models_path.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading docling models into:\n  {models_path}\n")
    print("This pulls ~300-500 MB and may take several minutes on first run.")
    print("Make sure you're on a network that allows huggingface.co.\n")

    download_models(
        output_dir=models_path,
        progress=True,
        with_layout=True,
        with_tableformer=True,
        with_picture_classifier=True,
        with_easyocr=True,
        with_code_formula=False,
        with_smolvlm=False,
        with_smoldocling=False,
        with_smoldocling_mlx=False,
        with_granite_vision=False,
    )

    print(f"\nDone. Models saved to: {models_path}")
    print("read_themes.py will now use these via config.DOCLING_MODELS_PATH.\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nDownload failed: {e}", file=sys.stderr)
        print("If this is an SSL / EOF error, you're still on the corp network.", file=sys.stderr)
        print("Switch to phone hotspot or home WiFi and retry.", file=sys.stderr)
        sys.exit(1)