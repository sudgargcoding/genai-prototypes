"""
read_themes.py — Read a theme PDF using docling (with OCR) and write a Markdown file.

Pipeline:
  • docling reads the PDF with OCR enabled and picture classification on.
    No LLM is used inside docling itself.
  • Page text (including OCR'd text from images) is written page-by-page.
  • Charts/graphs/maps detected by docling are described AFTER docling, by
    sending the picture image to the chat model.
  • Anything that cannot be read — empty pages, undescribable images,
    non-chart images — is flagged explicitly as "Unable to read this part".

Usage:
    python read_themes.py --pdf path/to/theme.pdf
    python read_themes.py --pdf path/to/theme.pdf --output out.md
"""

import truststore
truststore.inject_into_ssl()

import argparse
import base64
import io
import os
import re
from pathlib import Path

import boto3
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import PictureClassificationData

import config

# Picture types worth attempting to describe via the LLM.
CHART_PICTURE_TYPES = {"line_chart", "bar_chart", "pie_chart", "flow_chart", "map"}


# ---------------------------------------------------------------------------
# docling — OCR + picture classification, NO LLM inside
# ---------------------------------------------------------------------------

def _docling_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr                    = True
    pipeline_options.do_picture_classification = True
    pipeline_options.generate_picture_images   = True
    pipeline_options.images_scale              = 2

    # Force CPU — Apple Silicon MPS backend doesn't support float64, which
    # crashes docling's classifier/layout models with a MPS dtype error.
    pipeline_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU)

    # Use pre-downloaded models from DOCLING_MODELS_PATH if available, so docling
    # never tries to reach huggingface.co at runtime (blocked on corp network).
    # Run download_docling_models.py once on a non-corp network to populate this.
    if os.path.isdir(config.DOCLING_MODELS_PATH):
        pipeline_options.artifacts_path = config.DOCLING_MODELS_PATH
    else:
        print(
            f"  [warn] DOCLING_MODELS_PATH not found: {config.DOCLING_MODELS_PATH}\n"
            "         docling will attempt to download models from HuggingFace.\n"
            "         Run download_docling_models.py on a non-corp network first."
        )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


# ---------------------------------------------------------------------------
# used AFTER docling for chart description only
# ---------------------------------------------------------------------------

def _bedrock_client():
    return boto3.client("bedrock-runtime", region_name=config.BEDROCK_REGION)


def _describe_chart(image_bytes: bytes) -> str:
    """Ask chat model to describe a chart/graph image. Empty string on failure."""
    prompt = (
        "You are reading a chart from a financial publication. Be PRECISE.\n\n"
        "For any numeric values (axis ranges, band boundaries, data points):\n"
        "  • Quote only values you can clearly read in the image.\n"
        "  • If a value is unreadable, say so — DO NOT guess or interpolate.\n"
        "  • Distinguish 'I see a band from approximately X to Y' from "
        "    'a band labelled X to Y' — only use the latter when labels are visible.\n\n"
        "Then describe: chart type, what it measures, and any trends/key values.\n"
        "Keep to 3-5 sentences.\n"
        "If you cannot interpret the visual content at all, respond with exactly: UNABLE_TO_READ"
    )
    try:
        response = _bedrock_client().converse(
            modelId=config.CHAT_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"image": {"format": "png", "source": {"bytes": image_bytes}}},
                    {"text": prompt},
                ],
            }],
            inferenceConfig={"temperature": 0},
        )
        text = response["output"]["message"]["content"][0]["text"].strip()
        if not text or text.upper().startswith("UNABLE_TO_READ"):
            return ""
        return text
    except Exception as e:
        print(f"    [warn] LLM description failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# Picture helpers
# ---------------------------------------------------------------------------

def _picture_classification(picture) -> str | None:
    for ann in (picture.annotations or []):
        if isinstance(ann, PictureClassificationData):
            return ann.predicted_classes[0].class_name
    return None


def _picture_to_png_bytes(picture, doc) -> bytes | None:
    """Best-effort extraction of PNG bytes from a docling picture."""
    # Prefer the helper method when available
    try:
        pil_img = picture.get_image(doc)
        if pil_img is not None:
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        pass

    # Fall back to URI parsing (data:image/...;base64,... or file://...)
    try:
        if picture.image and picture.image.uri:
            uri = str(picture.image.uri)
            if uri.startswith("data:image"):
                m = re.match(r"data:image/[^;]+;base64,(.+)", uri)
                if m:
                    return base64.b64decode(m.group(1))
            elif uri.startswith("file://"):
                return Path(uri[len("file://"):]).read_bytes()
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def read_pdf(pdf_path: Path) -> str:
    converter = _docling_converter()
    print("  Reading with docling (OCR enabled) — this may take a minute on first run...")
    result = converter.convert(str(pdf_path))
    doc = result.document

    total_pages = len(doc.pages)
    print(f"  Pages : {total_pages}\n")

    # Group pictures by page
    pictures_by_page: dict[int, list] = {}
    for picture in doc.pictures:
        if picture.prov:
            page_no = picture.prov[0].page_no
            pictures_by_page.setdefault(page_no, []).append(picture)

    parts: list[str] = [f"# {pdf_path.stem}\n"]

    for page_no in sorted(doc.pages.keys()):
        print(f"  Page {page_no:>3}/{total_pages}", end="  |  ", flush=True)
        parts.append(f"\n\n---\n\n## Page {page_no} of {total_pages}\n\n")

        # ---- Page text (OCR'd by docling, no LLM) ----
        try:
            page_md = doc.export_to_markdown(
                page_no=page_no,
                include_annotations=False,
                image_placeholder="",
            ).strip()
        except Exception:
            page_md = ""

        if page_md:
            parts.append(page_md + "\n")
            print(f"{len(page_md)} text chars", end="", flush=True)
        else:
            parts.append("> **Unable to read this part** — no text could be extracted from this page.\n")
            print("no text", end="", flush=True)

        # ---- Pictures on this page ----
        page_pics = pictures_by_page.get(page_no, [])
        if page_pics:
            print(f"  |  {len(page_pics)} picture(s)", end="", flush=True)
            parts.append("\n\n### Visual Elements\n")
            for idx, picture in enumerate(page_pics, 1):
                classification = _picture_classification(picture)
                parts.append(
                    f"\n**Image {idx} on page {page_no}** "
                    f"(type: {classification or 'unclassified'})\n\n"
                )

                if classification not in CHART_PICTURE_TYPES:
                    parts.append(
                        "> **Unable to read this part** — image is not classified as a "
                        "chart, graph, or map; visual content cannot be extracted.\n"
                    )
                    continue

                image_bytes = _picture_to_png_bytes(picture, doc)
                if not image_bytes:
                    parts.append(
                        "> **Unable to read this part** — image data could not be extracted "
                        "from the document.\n"
                    )
                    continue

                description = _describe_chart(image_bytes)
                if description:
                    parts.append(description + "\n")
                else:
                    parts.append(
                        "> **Unable to read this part** — chart visual content could not "
                        "be interpreted.\n"
                    )

        print("  →  done")

    return "".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Read a theme PDF using docling (with OCR) and write a Markdown file. "
                    "Charts/graphs are described via Llm (post-docling). "
                    "Unreadable sections are flagged explicitly."
    )
    parser.add_argument("--pdf",    required=True, help="Path to the theme PDF (absolute or relative).")
    parser.add_argument("--output", default=None,
                        help="Output .md path. Defaults to <pdf-stem>.md next to the PDF.")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return

    output_path = (
        Path(args.output).expanduser().resolve() if args.output
        else pdf_path.with_suffix(".md")
    )

    print(f"\nInput  : {pdf_path}")
    print(f"Output : {output_path}\n")

    markdown = read_pdf(pdf_path)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"\nSaved  : {output_path}\n")


if __name__ == "__main__":
    main()
