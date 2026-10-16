# integrate_langfuse_observability

A small but complete **RAG pipeline over your own Markdown files**, with every LLM call traced in [Langfuse](https://langfuse.com). It shows what the pipeline sends to the model, what comes back, and how long it takes, so you can see where a bad answer came from.

Part of the [prototyping](../README.md) repository: quick, console-driven scripts for validating RAG and other GenAI concepts. It is exploratory code, not production software.

## What it validates

- How to ingest Markdown into a local vector store with sentence-level embeddings and parent-chunk context.
- How far query rewriting, query expansion, hybrid (semantic + keyword) retrieval and an LLM relevance filter improve answers.
- How to make answers **grounded**: answer only from retrieved context, surface contradictions between sources, and refuse when the context doesn't cover the question.
- How to add **observability** to a LangChain-based RAG flow with Langfuse, with minimal code and optional activation.

## Files

| File | Purpose |
|------|---------|
| `ingest.py` | Reads Markdown files, chunks and embeds them, and stores the vectors in ChromaDB |
| `search.py` | Takes a question, retrieves relevant chunks, and generates a grounded answer (with Langfuse tracing) |
| `config.py` | Paths, Bedrock region and model IDs, chunking, retrieval and Langfuse settings |
| `requirements.txt` | Python dependencies |

## How it works

### `ingest.py`: Markdown to vectors
1. Finds all `.md` / `.markdown` files under the docs folder.
2. Strips Markdown syntax, splits each file by heading, then into sentences.
3. Groups sentences into chunks within a token budget (default 512 tokens, with a 2-sentence overlap).
4. Embeds **each sentence**, prefixed with its document and section name, using a Bedrock embedding model.
5. Stores the vectors in a persistent, file-based [ChromaDB](https://www.trychroma.com/) collection (cosine distance). Each sentence vector keeps its **parent chunk** as the stored document, so retrieval returns readable context.
6. Keeps a manifest of file content hashes, so re-running only re-ingests files that changed. Use `--reset` to rebuild from scratch.

### `search.py`: question to grounded answer
1. **Rewrite** the question into a short declarative statement.
2. **Expand** it into three alternative phrasings to cover synonyms and terminology differences.
3. **Retrieve** with semantic search over the rewritten query and all alternatives, plus keyword search on acronyms and long terms. Results are de-duplicated and sorted by distance.
4. **Filter** by relevance: drop chunks beyond the distance threshold, then ask the LLM to keep only passages that are on the question's topic.
5. **Answer** using only the retrieved context. The prompt flags contradictions between sources, naming the files and sections involved. If the context isn't relevant, it refuses with a fixed "insufficient context" message instead of guessing.

```
question → rewrite → expand → semantic + keyword search → distance filter → topic filter → grounded answer
              └───────────────── each LLM call traced in Langfuse ─────────────────┘
```

## Observability with Langfuse

`search.py` attaches the Langfuse LangChain `CallbackHandler` to the chat model, so the query rewrite, query expansion, topic filter and answer generation calls are traced. Each trace shows the prompt, response, latency and token usage.

- Tracing is **optional**. It switches on only when both Langfuse keys are set. Without them, the script runs normally with no tracing.
- Only chat model calls are traced. Embedding calls and `ingest.py` are not.
- The script uses the Langfuse Python SDK **v3** API (`from langfuse.langchain import CallbackHandler`).

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python **3.13** | The version the prototype is developed with |
| AWS account with **Amazon Bedrock** access | Model access enabled for your chat and embedding models in your chosen region |
| AWS credentials | Configured locally, for example `aws configure` or `aws sso login`, or via environment variables |
| Langfuse project (optional) | [Langfuse Cloud](https://cloud.langfuse.com) or self-hosted, with public and secret keys |
| Some Markdown files | Anything you'd like to ask questions about |

## Setup

```bash
cd integrate_langfuse_observability
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # chromadb, tiktoken, langchain-aws, langfuse, boto3
```

### Configure (`config.py`)

Set the AWS region and models for your account. These are read from environment variables, with empty defaults in `config.py`, so you must set them:

```bash
export BEDROCK_REGION="us-east-1"                   # region where Bedrock models are enabled
export CHAT_MODEL="<your-chat-model-id>"            # model or inference profile for generation
export EMBED_MODEL="amazon.titan-embed-text-v2:0"   # model for embeddings
```

To enable tracing, add your Langfuse keys:

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_BASE_URL="https://cloud.langfuse.com"   # default, change for self-hosted or another region
```

Where your documents and vectors live is set by two plain paths in `config.py`: `DOCS_DIR` (default `../my_docs`) and `VECTORS_DIR` (default `../my_vectors`). Both resolve to folders next to this one, in `prototyping/`. Create `my_docs/` and add your Markdown files, or edit the paths.

Notes:
- Model availability differs by region. If a call fails with an access or "model not found" error, enable the model in the Bedrock console for that region.
- Use the same embedding model for ingest and search. Vectors from different embedding models can't be compared.
- Don't commit credentials. AWS access uses the standard credential chain, and the Langfuse keys come from the environment.

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

Then open your Langfuse project to inspect the traces.

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
| | `--top-k` | 10 | Chunks to retrieve |
| | `--no-generate` | off | Show retrieved chunks only |

## Settings in `config.py`

| Variable | Default | Purpose |
|----------|---------|---------|
| `BEDROCK_REGION`, `EMBED_MODEL`, `CHAT_MODEL` | empty, you must set them | AWS region and model IDs |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | 512, 2 | Chunking |
| `COLLECTION_NAME` | `my_docs` | ChromaDB collection |
| `DISTANCE_THRESHOLD` | 0.9 | Cosine distance cut-off (0 = identical, 2 = opposite). Chunks farther than this are dropped before answering |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | empty (tracing off) | Langfuse credentials |
| `LANGFUSE_BASE_URL` | `https://cloud.langfuse.com` | Langfuse host |

All of these can be set through environment variables. `DOCS_DIR` and `VECTORS_DIR` are plain paths in the file.

## Costs and safety

- Bedrock calls are **billed per use**. One question makes several LLM calls (rewrite, expand, filter, answer) plus several embedding calls. Start with a small set of documents.
- Langfuse stores the **full prompts and responses**, including the retrieved text from your documents. Don't use sensitive or customer data unless you're comfortable with where Langfuse is hosted and who can see the project.
- `my_docs/` and `my_vectors/` sit outside this folder. Don't commit them.

## Known limitations

- `requirements.txt` pins `langfuse>=2.0`, but the code needs the v3 SDK. Use `langfuse>=3`.
- Retrieval quality depends heavily on the prompts and the `DISTANCE_THRESHOLD`, which are not tuned or evaluated here.
