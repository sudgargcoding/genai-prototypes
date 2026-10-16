# rag-pipeline

A compact, readable **RAG (Retrieval-Augmented Generation) pipeline over your own Markdown files**. It ingests documents into a local vector store using **sentence-based chunks** and Bedrock embeddings. It retrieves with **cosine similarity and k-nearest-neighbour (kNN) search**, and answers only from what it retrieved, refusing when the context doesn't cover the question.

Part of the [genai-prototypes](../README.md) repository: quick, console-driven scripts for validating RAG and other GenAI concepts. It is exploratory code, not production software.

> **Background reading:** this prototype is the hands-on companion to the article **[0 → 1 with RAG](https://github.com/sudgargcoding/technical-writeups/blob/main/rag-zero-to-one/article.md)** in the [technical-writeups](https://github.com/sudgargcoding/technical-writeups) repository. The article explains the terminology, the building blocks, safety and governance, evaluation metrics and failure scenarios. This code lets you try the ideas.

For the same pipeline with LLM tracing added, see [`rag-langfuse-observability`](../rag-langfuse-observability/README.md).

## What it validates

- **Sentence-based chunking** with a token budget and sentence overlap, so chunk boundaries never cut a sentence in half.
- **Cosine similarity + kNN retrieval** over embeddings, with a distance cut-off to discard weak matches.
- Whether **query rewriting, query expansion and keyword matching** improve retrieval over a plain nearest-neighbour search.
- **Grounded generation:** answer only from retrieved context, flag contradictions between sources, and refuse instead of guessing.

## Key concepts

These match the terms in the [0 → 1 with RAG](https://github.com/sudgargcoding/technical-writeups/blob/main/rag-zero-to-one/article.md) article.

| Concept | How this prototype uses it |
|---------|----------------------------|
| **Embedding** | Each sentence (and each query) is turned into a vector by a Bedrock embedding model |
| **Cosine similarity / distance** | The ChromaDB collection is created with cosine distance (`hnsw:space = cosine`). Distance 0 means identical direction, and 2 means opposite. Results are shown with similarity = (1 − distance) × 100% |
| **kNN** | Retrieval asks the vector store for the `k` nearest neighbours of the query vector (`--top-k`, default 10). ChromaDB uses an HNSW index, so the search is *approximate* nearest neighbour (ANN) |
| **Sentence-based chunking** | Text is split into sentences, and consecutive sentences are grouped into chunks of up to 512 tokens, with a 2-sentence overlap between neighbouring chunks |
| **Parent-chunk retrieval** | Every sentence is embedded separately for precise matching, but the stored document is the sentence's whole parent chunk, so the LLM gets surrounding context |
| **Relevance threshold** | Chunks with cosine distance above `DISTANCE_THRESHOLD` (default 0.9) are dropped before answering |
| **Grounding** | The answer prompt allows only the retrieved context, and a fixed refusal message is returned when nothing relevant is found |

## Files

| File | Purpose |
|------|---------|
| `ingest.py` | Reads Markdown files, chunks and embeds them, and stores the vectors in ChromaDB |
| `search.py` | Takes a question, retrieves relevant chunks, and generates a grounded answer |
| `config.py` | Paths, Bedrock region and model IDs, chunking and retrieval settings |
| `requirements.txt` | Python dependencies (`chromadb`, `tiktoken`, `langchain-aws`, `boto3`) |

## How it works

### Ingestion (`ingest.py`): Markdown to vectors
1. Finds all `.md` / `.markdown` files under the docs folder.
2. Strips Markdown syntax (code blocks, links, emphasis, lists, headings), then splits each file by heading, and each section into sentences.
3. Groups sentences into **sentence-based chunks** within the token budget (counted with `tiktoken`), overlapping by a few sentences.
4. Embeds **each sentence**, prefixed with its document and section name, using a Bedrock embedding model.
5. Stores the vectors in a persistent, file-based [ChromaDB](https://www.trychroma.com/) collection with cosine distance. Metadata keeps the source file, section, chunk and sentence index.
6. Keeps a manifest of file content hashes, so re-running only re-ingests files that changed. Use `--reset` to rebuild from scratch.

### Query (`search.py`): question to grounded answer
1. **Rewrite** the question into a short declarative statement.
2. **Expand** it into three alternative phrasings to cover synonyms and terminology differences.
3. **Retrieve:** embed the rewritten query and each alternative, run a **kNN search by cosine distance** for each, and add a keyword search on acronyms and long terms. Results are de-duplicated and sorted by distance, and the best `top-k` are kept.
4. **Filter:** drop chunks beyond the distance threshold, then ask the LLM to keep only the passages that are on the question's topic.
5. **Answer** using only the retrieved context. The prompt flags contradictions between sources (naming the files and sections involved). If the context isn't relevant, it replies with a fixed "insufficient context" message instead of guessing.

```
Ingest:  Markdown → strip → sections → sentences → sentence-based chunks → embed each sentence → ChromaDB (cosine)

Query:   question → rewrite → expand → embed → kNN (cosine) + keyword → distance filter → topic filter → grounded answer
```

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python **3.13** | The version the prototype is developed with |
| AWS account with **Amazon Bedrock** access | Model access enabled for your chat and embedding models in your chosen region |
| AWS credentials | Configured locally, for example `aws configure` or `aws sso login`, or via environment variables |
| Some Markdown files | Anything you'd like to ask questions about. The articles in the [technical-writeups](https://github.com/sudgargcoding/technical-writeups) repository make a good test set |

## Setup

```bash
cd rag-pipeline
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure (`config.py`)

Set the AWS region and models for your account. They are read from environment variables, with empty defaults in `config.py`, so you must set them:

```bash
export BEDROCK_REGION="us-east-1"                   # region where Bedrock models are enabled
export CHAT_MODEL="<your-chat-model-id>"            # model or inference profile for generation
export EMBED_MODEL="amazon.titan-embed-text-v2:0"   # model for embeddings
```

Where your documents and vectors live is set by two plain paths in `config.py`: `DOCS_DIR` (default `../my_docs`) and `VECTORS_DIR` (default `../my_vectors`). Both resolve to folders next to this one, in `genai-prototypes/`. Create `my_docs/` and add your Markdown files, or edit the paths.

Notes:
- Model availability differs by region. If a call fails with an access or "model not found" error, enable the model in the Bedrock console for that region.
- Use the same embedding model for ingest and search. Vectors from different embedding models can't be compared.
- Don't commit credentials. AWS access uses the standard credential chain, so there is no AWS key in `config.py`.

## Run

```bash
# 1. Build or update the vector store (default: ../my_vectors)
python ingest.py
python ingest.py --reset          # rebuild from scratch

# 2. Ask a question: prints the retrieved chunks, then the grounded answer
python search.py "What are the main challenges of sports betting platforms?"

# Retrieval only, skipping answer generation
python search.py "your question" --no-generate
```

Retrieved chunks are printed with their source file, section, cosine distance and similarity, so you can see why each one was selected.

## Command-line options

| Script | Option | Default | Meaning |
|--------|--------|---------|---------|
| `ingest.py` | `--docs-dir` | `config.DOCS_DIR` | Folder with Markdown files |
| | `--vectors-dir` | `config.VECTORS_DIR` | Where ChromaDB is stored |
| | `--embed-model` | `config.EMBED_MODEL` | Bedrock embedding model ID |
| | `--chunk-size` | 512 | Tokens per chunk |
| | `--chunk-overlap` | 2 | Sentences shared between chunks |
| | `--collection` | `config.COLLECTION_NAME` | ChromaDB collection name |
| | `--reset` | off | Delete the collection and re-ingest |
| `search.py` | `query` (positional) | none | The question |
| | `--vectors-dir`, `--collection`, `--embed-model` | from `config.py` | Must match the values used at ingest time |
| | `--chat-model` | `config.CHAT_MODEL` | Bedrock chat model ID |
| | `--top-k` | 10 | Number of nearest neighbours (chunks) to retrieve |
| | `--no-generate` | off | Show retrieved chunks only |

## Settings in `config.py`

| Variable | Default | Purpose |
|----------|---------|---------|
| `BEDROCK_REGION`, `EMBED_MODEL`, `CHAT_MODEL` | empty, you must set them | AWS region and model IDs |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | 512, 2 | Chunk size in tokens, and sentence overlap |
| `COLLECTION_NAME` | `my_docs` | ChromaDB collection |
| `DISTANCE_THRESHOLD` | 0.9 | Cosine distance cut-off (0 = identical, 2 = opposite). Chunks farther than this are dropped before answering |

All of these can be set through environment variables. `DOCS_DIR` and `VECTORS_DIR` are plain paths in the file.

## Experiments to try

These connect to the "common failure scenarios" and evaluation sections of the [article](https://github.com/sudgargcoding/technical-writeups/blob/main/rag-zero-to-one/article.md).

- Change `--chunk-size` and `--chunk-overlap`, re-ingest with `--reset`, and compare the retrieved chunks.
- Lower and raise `DISTANCE_THRESHOLD` to see the trade-off between missing relevant context and letting in noise.
- Run `--no-generate` to judge retrieval quality on its own, for example with Recall@k and Precision@k on a small question set.
- Ask about something your documents don't cover, and confirm you get the refusal message and not an invented answer.
- Put two documents with contradicting statements into `my_docs/` and check that the answer flags the conflict.

## Costs and safety

- Bedrock calls are **billed per use**. One question makes several LLM calls (rewrite, expand, filter, answer) plus several embedding calls. Start with a small set of documents.
- Your document text and questions are sent to the Bedrock models. Don't use sensitive or customer data unless you understand where it goes.
- `my_docs/` and `my_vectors/` sit outside this folder. Don't commit them.

## Known limitations

- Retrieval quality depends heavily on the prompts and the `DISTANCE_THRESHOLD`, which are not tuned or evaluated here.
- The vector store is local and file-based, which is fine for experiments but not for multi-user or large-scale use.
