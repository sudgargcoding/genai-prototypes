# prototyping

Small, quick-to-run code prototypes for validating **RAG (Retrieval-Augmented Generation)** and other modern GenAI and architecture concepts hands-on.

The articles in my [docs](https://github.com/sudgargcoding/docs) repository discuss these ideas. This repository is where I try them out in code, before or while writing about them. The goal is a fast answer to "does this actually work, and how?", not production-ready software.

> **Work in progress.** Prototypes are added as I explore new ideas, so the collection will keep growing.

## Prototypes

Each prototype lives in its own folder with a README that covers what it validates, how it works, and how to set it up and run it.

| Prototype | What it validates |
|-----------|-------------------|
| [`rag_pipeline`](rag_pipeline/README.md) | A baseline RAG pipeline over local Markdown files: sentence-based chunking, embeddings in ChromaDB, cosine-similarity kNN retrieval with query rewriting, expansion and keyword matching, and grounded answers that flag conflicts or refuse. Companion to the article [0 → 1 with RAG](https://github.com/sudgargcoding/docs/blob/main/0to1withRAG/article.md). |
| [`integrate_langfuse_observability`](integrate_langfuse_observability/README.md) | An end-to-end RAG pipeline over local Markdown files (the `rag_pipeline` approach) with LLM calls traced in **Langfuse**. |
| [`ocr_docling`](ocr_docling/README.md) | Reading PDFs, including scanned pages and charts, with **Docling** OCR and writing Markdown. Charts are described by a Bedrock chat model, and anything unreadable is flagged explicitly. |

## How the repository is organised

- **Scripts, not applications.** Each prototype is a set of standalone Python scripts that you launch from the console.
- **One folder per prototype.** Each folder is self-contained, with its own `config.py`, its own README, and its own `requirements.txt` where it needs one. Run the scripts from inside that folder.
- **Quick to read and change.** Prototypes are intentionally small and light on abstraction, so you can see the whole idea in one sitting and modify it.
- **Exploratory quality.** Error handling is minimal, and the focus is on validating concepts rather than hardening them. Don't deploy these as they are.

```
prototyping/
├── rag_pipeline/                       # baseline RAG: sentence chunks, cosine kNN, grounded answers
├── integrate_langfuse_observability/   # the same RAG pipeline with Langfuse observability
├── ocr_docling/                        # PDF OCR to Markdown with Docling
└── README.md
```

New prototypes follow the same pattern: a folder, a `config.py`, a README, and a row in the table above.

## Common ground

Details differ per prototype, so check each folder's README. What they share:

- **Python 3.13**, in a virtual environment per prototype.
- **AWS Bedrock** for chat and embedding models. You need an AWS account with Bedrock access, with model access enabled in your chosen region, and AWS credentials configured locally (for example with `aws configure` or `aws sso login`). Set the region and model IDs in each prototype's `config.py`, or through the environment variables it reads.
- Typical topics include RAG building blocks (chunking, embeddings, vector search, retrieval and reranking, grounded answer generation, evaluation, observability) and other emerging GenAI patterns.

```bash
git clone git@github.com:sudgargcoding/prototyping.git
cd prototyping/<prototype-folder>
python3.13 -m venv .venv && source .venv/bin/activate
# then follow that folder's README
```

## Costs and safety

- Bedrock calls (chat and embeddings) are **billed per use**. Start with small inputs and clear the resources you create.
- Don't point prototypes at sensitive or customer data unless you understand where it is sent and stored. Some prototypes send content to external services such as Bedrock or Langfuse.
- Don't commit credentials. Scripts use the standard AWS credential chain, and other secrets belong in environment variables.

## Related

- [docs](https://github.com/sudgargcoding/docs): the articles these experiments accompany, including *0 → 1 with RAG*.

## Feedback

This is a personal learning repository. Suggestions, corrections and ideas for new prototypes are welcome. Open an issue or a pull request.
