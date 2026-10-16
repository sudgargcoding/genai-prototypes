# ocr_docling

Reads a **PDF with OCR using [Docling](https://github.com/docling-project/docling)** and writes it out as **Markdown**, page by page. Charts, graphs and maps are described afterwards by a vision-capable Bedrock chat model. Anything that can't be read is flagged explicitly instead of being guessed.

Part of the [prototyping](../README.md) repository: quick, console-driven scripts for validating RAG and other GenAI concepts. It is exploratory code, not production software.

## What it validates

- Whether Docling's OCR, layout detection and picture classification can turn real-world PDFs (including scanned pages) into clean Markdown, suitable as input for a RAG pipeline.
- Whether a Bedrock chat model can describe charts without inventing numbers.
- A "never silently skip" rule: every page or image that can't be read is marked **"Unable to read this part"**, so gaps are visible to whoever reads the output.

## Files

| File | Purpose |
|------|---------|
| `read_themes.py` | Main script: converts one PDF to Markdown |
| `download_docling_models.py` | One-time download of the Docling models for offline use |
| `config.py` | Bedrock region and chat model, and the path to the Docling models |

## How it works

Docling does the reading, and the LLM is used only for charts.

1. **Docling reads the PDF** with OCR on and picture classification on. No LLM is involved at this stage. It runs on CPU, because Apple Silicon's MPS backend crashes on Docling's classifier and layout models.
2. **Page text** (including text recognised from images) is written page by page under `## Page N of M` headings.
3. **Pictures** are classified by Docling. Only charts, graphs and maps (`line_chart`, `bar_chart`, `pie_chart`, `flow_chart`, `map`) are sent on.
4. **Chart description:** each such picture goes to the Bedrock chat model (Converse API, temperature 0) with a strict prompt. It may quote only values it can clearly read, must not guess or interpolate, and replies `UNABLE_TO_READ` if it can't interpret the image. The answer is limited to 3–5 sentences.
5. **Flagging:** pages with no text, images that aren't charts, images whose data can't be extracted, and charts the model can't interpret are each written as a `> **Unable to read this part** — <reason>` note.

```
PDF → Docling (OCR + layout + picture classification) → page text
                                                      → charts → Bedrock chat model → description
                                                      → anything unreadable → "Unable to read this part"
                                                      → Markdown file
```

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python **3.13** | The version the prototype is developed with |
| AWS account with **Amazon Bedrock** access | A chat model that accepts images, enabled in your chosen region |
| AWS credentials | Configured locally, for example `aws configure` or `aws sso login`, or via environment variables |
| Network access to `huggingface.co` (once) | Only for the one-time model download |
| Disk space | About 300–500 MB for the Docling models |

## Setup

This prototype has no `requirements.txt`. Install the packages it imports:

```bash
cd ocr_docling
python3.13 -m venv .venv
source .venv/bin/activate
pip install docling docling-core boto3 truststore
```

### Configure (`config.py`)

Edit these values in `config.py`:

| Setting | Meaning |
|---------|---------|
| `BEDROCK_REGION` | AWS region where Bedrock is enabled, for example `us-east-1` |
| `CHAT_MODEL` | Bedrock model ID or inference profile ID for describing charts. It must support image input |
| `DOCLING_MODELS_PATH` | Where the Docling models live. Defaults to `.docling-models/` in this folder, or set the `DOCLING_MODELS_PATH` environment variable |

Notes:
- `BEDROCK_REGION` and `CHAT_MODEL` are empty in the file and are plain values, not environment variables.
- Model availability differs by region. If calls fail with an access or "model not found" error, enable the model in the Bedrock console for that region.
- Don't commit credentials. AWS access uses the standard credential chain.

### Download the Docling models (once)

```bash
python download_docling_models.py
python download_docling_models.py --models-path /custom/dir   # optional
```

This pulls the models the script actually uses (layout, table structure, picture classifier, EasyOCR) and skips the heavy optional ones. Run it on a network that allows `huggingface.co`, such as a phone hotspot or home Wi-Fi. Afterwards `read_themes.py` loads the models from disk and never contacts Hugging Face at runtime. If the models folder is missing, the script warns and tries to download them itself.

## Run

```bash
python read_themes.py --pdf path/to/document.pdf
python read_themes.py --pdf path/to/document.pdf --output out.md
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--pdf` | required | Path to the PDF (absolute or relative) |
| `--output` | `<pdf-name>.md` next to the PDF | Where to write the Markdown |

The script prints progress per page (text size, number of pictures) and saves the Markdown file at the end. The first run is slower, because Docling loads its models.

### Output shape

```markdown
# <pdf name>

---

## Page 1 of 12

<page text from OCR>

### Visual Elements

**Image 1 on page 1** (type: bar_chart)

<3–5 sentence description of the chart, or an "Unable to read this part" note>
```

## Costs and safety

- Chart descriptions call Bedrock, which is **billed per use**, once per chart picture. Try a short PDF first.
- Chart images and their text are sent to the Bedrock model. Don't use confidential PDFs unless you understand where that data goes.
- Chart descriptions come from an LLM. Treat numbers in them as unverified and compare them against the source PDF.

## Known limitations

- The chart prompt is written for financial publications, so descriptions may suit other kinds of charts less well.
- Only the picture types listed above are described. All other images are flagged as unreadable, even if they contain useful text or diagrams.
- The converter is forced to CPU, which is slower on large PDFs.
- `truststore` is imported so Python uses the operating system's certificate store, which helps on networks with a corporate proxy.
- `config.py` also defines `THEMES_DIR`, which the script doesn't currently use.
