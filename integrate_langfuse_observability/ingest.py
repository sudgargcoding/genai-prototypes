"""
ingest.py — Scan markdown files, chunk them, embed
and persist vectors to vector Db under ../my_vectors.
"""

import os
import re
import sys
import json
import hashlib
import argparse
from pathlib import Path

import chromadb
import tiktoken
from langchain_aws import BedrockEmbeddings

import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenizer():
    """Return a tiktoken encoder used only for *counting* tokens."""
    # cl100k_base is a reasonable proxy for any modern LLM tokeniser
    return tiktoken.get_encoding("cl100k_base")


def chunk_sentences(sentences: list[str], chunk_size: int, overlap_sents: int) -> list[list[str]]:
    """Group complete sentences into token-budget chunks with sentence-level overlap."""
    enc = _tokenizer()
    token_counts = [len(enc.encode(s)) for s in sentences]
    chunks = []
    start = 0
    while start < len(sentences):
        group: list[str] = []
        total = 0
        i = start
        while i < len(sentences):
            if total + token_counts[i] > chunk_size and group:
                break
            group.append(sentences[i])
            total += token_counts[i]
            i += 1
        if group:
            chunks.append(group)
        if i >= len(sentences):
            break
        start = max(start + 1, i - overlap_sents)
    return chunks


def doc_id(file_path: str, chunk_index: int, sentence_index: int = 0) -> str:
    """Stable, unique ID for a sentence within a chunk."""
    raw = f"{file_path}::{chunk_index}::{sentence_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def file_hash(text: str) -> str:
    """SHA256 of file content — used to detect edits."""
    return hashlib.sha256(text.encode()).hexdigest()


def _manifest_path(vectors_dir: str) -> str:
    return os.path.join(vectors_dir, "file_manifest.json")


def load_manifest(vectors_dir: str) -> dict:
    path = _manifest_path(vectors_dir)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_manifest(vectors_dir: str, manifest: dict) -> None:
    with open(_manifest_path(vectors_dir), "w") as f:
        json.dump(manifest, f, indent=2)


def strip_markdown(text: str) -> str:
    """Remove markdown syntax, keeping only prose text for cleaner embeddings."""
    text = re.sub(r'```[\s\S]*?```', '', text)           # fenced code blocks
    text = re.sub(r'`[^`\n]+`', '', text)                 # inline code
    text = re.sub(r'!\[([^\]]*)\]\([^\)]*\)', r'\1', text) # images
    text = re.sub(r'\[([^\]]+)\]\([^\)]*\)', r'\1', text)  # links
    text = re.sub(r'\*{1,3}([^*\n]+)\*{1,3}', r'\1', text) # bold/italic
    text = re.sub(r'_{1,3}([^_\n]+)_{1,3}', r'\1', text)   # underscore emphasis
    text = re.sub(r'^>\s?', '', text, flags=re.MULTILINE)   # blockquotes
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)  # hr
    text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)  # bullet lists
    text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)  # numbered lists
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)       # headings
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Split stripped text into individual sentences for fine-grained embedding."""
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z\d])', text)
    return [s.strip() for s in parts if len(s.strip()) >= 15]


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs, preserving section context."""
    heading_re = re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE)
    positions = [(m.start(), m.group(1).strip(), m.end()) for m in heading_re.finditer(text)]

    if not positions:
        return [("", text)]

    sections = []
    if positions[0][0] > 0:
        preamble = text[:positions[0][0]].strip()
        if preamble:
            sections.append(("", preamble))

    for i, (_, heading, body_start) in enumerate(positions):
        body_end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        body = text[body_start:body_end].strip()
        sections.append((heading, body))

    return sections


def embed(texts: list[str], model: str) -> list[list[float]]:
    """Batch-embed a list of texts using AWS Bedrock."""
    client = BedrockEmbeddings(model_id=model, region_name=config.BEDROCK_REGION)
    return client.embed_documents(texts)


def load_markdown_files(docs_dir: str) -> list[dict]:
    """Recursively find all .md / .markdown files and return raw records."""
    root = Path(docs_dir)
    records = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in {".md", ".markdown"} and path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            records.append({"path": str(path), "text": text})
    return records


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------

def ingest(docs_dir: str, vectors_dir: str, embed_model: str,
           chunk_size: int, overlap: int, collection_name: str,
           reset: bool = False):

    docs_dir = os.path.abspath(docs_dir)
    vectors_dir = os.path.abspath(vectors_dir)
    os.makedirs(vectors_dir, exist_ok=True)

    # --- ChromaDB (persistent, file-based) ---
    chroma = chromadb.PersistentClient(path=vectors_dir)
    if reset:
        try:
            chroma.delete_collection(collection_name)
            print(f"[reset] Deleted existing collection '{collection_name}'.")
        except Exception:
            pass
    collection = chroma.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # --- Discover docs ---
    records = load_markdown_files(docs_dir)
    if not records:
        print(f"No markdown files found in {docs_dir}.")
        return

    print(f"Found {len(records)} markdown file(s) in {docs_dir}")

    # --- Manifest: tracks content hash per file to detect edits ---
    manifest = {} if reset else load_manifest(vectors_dir)

    total_new = 0
    for rec in records:
        rel_path = os.path.relpath(rec["path"], docs_dir)
        content_hash = file_hash(rec["text"])

        if manifest.get(rel_path) == content_hash:
            print(f"  {rel_path}: unchanged, skipped")
            continue

        # Build sentence-level entries: embed each sentence, store parent chunk as document.
        # Strip and split sentences first so chunk boundaries always fall between sentences.
        sections = split_into_sections(rec["text"])
        # entries: (parent_chunk_text, embed_text, sentence, heading, chunk_idx, sent_idx)
        entries: list[tuple[str, str, str, str, int, int]] = []
        chunk_counter = 0
        for heading, body in sections:
            if not body.strip():
                continue
            prefix = f"Document: {rel_path}"
            if heading:
                prefix += f" | Section: {heading}"
            sentences = split_sentences(strip_markdown(body))
            if not sentences:
                continue
            for sent_group in chunk_sentences(sentences, chunk_size, overlap):
                parent_text = " ".join(sent_group)
                for sent_idx, sentence in enumerate(sent_group):
                    entries.append((parent_text, f"{prefix}\n\n{sentence}", sentence, heading, chunk_counter, sent_idx))
                chunk_counter += 1

        if not entries:
            continue

        print(f"  {rel_path}: {chunk_counter} chunk(s) → {len(entries)} sentence(s)", end="", flush=True)

        # Delete stale entries for this file before re-ingesting
        try:
            stale = collection.get(where={"source": rel_path}, include=[])
            if stale["ids"]:
                collection.delete(ids=stale["ids"])
                print(f"  (removed {len(stale['ids'])} stale)", end="", flush=True)
        except Exception:
            pass

        ids            = [doc_id(rec["path"], ci, si) for *_, ci, si in entries]
        metas          = [
            {"source": rel_path, "chunk_index": ci, "sentence_index": si,
             "total_chunks": chunk_counter, "section": heading, "sentence": sent}
            for _, _, sent, heading, ci, si in entries
        ]
        original_texts = [e[0] for e in entries]
        embed_texts    = [e[1] for e in entries]

        vectors = embed(embed_texts, embed_model)
        # Embed sentence-level text; store parent chunk for readable retrieval
        collection.upsert(ids=ids, embeddings=vectors, documents=original_texts, metadatas=metas)

        manifest[rel_path] = content_hash
        total_new += len(entries)
        print(f"  → {len(entries)} sentence vector(s) added")

    save_manifest(vectors_dir, manifest)
    print(f"\nDone. {total_new} new/updated sentence vector(s) stored in {vectors_dir}")
    print(f"Collection '{collection_name}' now has {collection.count()} chunk(s) total.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest markdown docs into a local ChromaDB vector store.")
    parser.add_argument("--docs-dir",         default=config.DOCS_DIR,        help="Directory containing markdown files")
    parser.add_argument("--vectors-dir",      default=config.VECTORS_DIR,     help="Directory to persist ChromaDB")
    parser.add_argument("--embed-model",      default=config.EMBED_MODEL,     help="model for embeddings")
    parser.add_argument("--chunk-size",       default=config.CHUNK_SIZE,      type=int, help="Tokens per chunk")
    parser.add_argument("--chunk-overlap",    default=config.CHUNK_OVERLAP,   type=int, help="Number of sentences to overlap between chunks")
    parser.add_argument("--collection",       default=config.COLLECTION_NAME, help="ChromaDB collection name")
    parser.add_argument("--reset",            action="store_true",            help="Delete the collection and re-ingest from scratch")
    args = parser.parse_args()

    ingest(
        docs_dir=args.docs_dir,
        vectors_dir=args.vectors_dir,
        embed_model=args.embed_model,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
        collection_name=args.collection,
        reset=args.reset,
    )


if __name__ == "__main__":
    main()
