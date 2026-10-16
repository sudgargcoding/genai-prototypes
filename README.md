# prototyping

Small, quick-to-run code prototypes for validating **RAG (Retrieval-Augmented Generation)** and other modern GenAI / architecture concepts hands-on.

The articles in my [docs](https://github.com/sudgargcoding/docs) repository discuss these ideas. This repository is where I try them out in code before (or while) writing about them: the goal is a fast answer to "does this actually work, and how?", not production-ready software.

> **Work in progress.** Prototypes are added as I explore new ideas, so the collection will keep growing.

## What to expect

- **Scripts, not applications.** Each prototype is a standalone Python script that you launch from the console.
- **Quick to read and change.** Prototypes are intentionally small and light on abstraction, so you can see the whole idea in one sitting and modify it.
- **Shared configuration.** Settings common to all scripts (AWS Bedrock region and model IDs) live in one place: `config.py`.
- **Exploratory quality.** Error handling is minimal, and the focus is on validating concepts rather than hardening them. Don't deploy these as they are.

Typical topics include RAG building blocks (chunking, embeddings, vector search, retrieval and reranking, grounded answer generation, evaluation) and other emerging GenAI patterns.

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python **3.13** | The version the prototypes are developed and tested with |
| AWS account with **Amazon Bedrock** access | Model access must be enabled for the chat and embedding models you choose, in your chosen region |
| AWS credentials | Configured locally, for example with `aws configure` or `aws sso login`, or via environment variables |

## Setup

```bash
git clone git@github.com:sudgargcoding/prototyping.git
cd prototyping

python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # if the prototype you want to run ships one
```

## Configuration (`config.py`)

Before running anything, open `config.py` and set the following to match your AWS account:

1. **Bedrock region:** the AWS region where you have Bedrock enabled (for example `us-east-1`).
2. **Chat / LLM model:** the Bedrock model ID, or inference profile ID, used for answer generation.
3. **Embedding model:** the Bedrock model ID used to create vector embeddings.

An illustrative sketch, with variable names that may differ slightly in the actual file:

```python
# config.py
BEDROCK_REGION = "us-east-1"                       # region where Bedrock models are enabled
CHAT_MODEL_ID = "<your-chat-model-id>"             # model or inference profile for generation
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"  # model for embeddings
```

Notes:
- Model availability differs by region. If a call fails with an access or "model not found" error, check that the model is enabled in the Bedrock console for that region.
- Keep the chat and embedding choices consistent within a run. Vectors produced by one embedding model can't be searched with vectors from another.
- Don't commit credentials. Scripts use the standard AWS credential chain, so there is no key in `config.py`.

## Running a prototype

From the repository root, with the virtual environment active:

```bash
python <script_name>.py
```

Each script's header comment or `--help` output (where provided) describes what it demonstrates, which inputs it expects and what it prints.

## Repository layout

```
prototyping/
├── config.py        # Bedrock region and model IDs shared by all scripts
├── <prototype>.py   # one self-contained script per idea
└── README.md
```

This layout will be expanded, and a table of prototypes added here, as the first scripts land.

## Costs and safety

- Bedrock calls (chat and embeddings) are **billed per use**. Start with small inputs and clear the resources you create.
- Don't point prototypes at sensitive or customer data unless you understand where it is sent and stored.

## Related

- [docs](https://github.com/sudgargcoding/docs): the articles these experiments accompany, including *0 → 1 with RAG*.

## Feedback

This is a personal learning repository. Suggestions, corrections and ideas for new prototypes are welcome. Open an issue or a pull request.
