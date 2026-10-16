"""
search.py — Query the ChromaDB vector store and optionally generate an answer
using chat model.
"""

import argparse
import os
import textwrap

import chromadb
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from langchain_core.messages import HumanMessage

import config


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def get_collection(vectors_dir: str, collection_name: str):
    vectors_dir = os.path.abspath(vectors_dir)
    chroma = chromadb.PersistentClient(path=vectors_dir)
    return chroma.get_collection(name=collection_name)


def embed_query(query: str, model: str) -> list[float]:
    client = BedrockEmbeddings(model_id=model, region_name=config.BEDROCK_REGION)
    return client.embed_query(query)


def rewrite_query(query: str, chat_model: str) -> str:
    """Rewrite the question as a declarative search statement without inventing facts."""
    llm = ChatBedrockConverse(model_id=chat_model, region_name=config.BEDROCK_REGION)
    prompt = (
        "Rewrite the following question as a short declarative statement (1-2 sentences) "
        "that describes what information is being sought. "
        "Keep all key terms and named entities. "
        "Do NOT add any facts, details, or claims — only rephrase the intent.\n\n"
        f"Question: {query}\n\nStatement:"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def key_terms(query: str) -> list[str]:
    """Extract terms worth keyword-searching: acronyms and words 4+ chars."""
    import re
    words = re.findall(r"[A-Za-z0-9]+", query)
    return list({w for w in words if w.isupper() or len(w) >= 4})


def expand_query(query: str, chat_model: str, n: int = 3) -> list[str]:
    """Generate alternative phrasings to handle synonyms and terminology differences."""
    llm = ChatBedrockConverse(model_id=chat_model, region_name=config.BEDROCK_REGION)
    prompt = (
        f"Generate {n} alternative ways to express the following query using different "
        f"terminology or synonyms that might appear in documents. "
        f"Preserve the original intent but vary the wording (e.g. formal/informal terms, "
        f"abbreviations, alternative names). "
        f"Return ONLY the alternatives, one per line, no numbering or explanation.\n\n"
        f"Query: {query}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return [line.strip() for line in response.content.strip().splitlines() if line.strip()][:n]


def _vector_search(vector: list[float], collection, top_k: int) -> list[tuple]:
    results = collection.query(
        query_embeddings=[vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return list(zip(results["documents"][0], results["metadatas"][0], results["distances"][0]))


def retrieve(query: str, collection, embed_model: str, chat_model: str, top_k: int) -> tuple[list[dict], str]:
    rewritten = rewrite_query(query, chat_model)
    alternatives = expand_query(query, chat_model)

    seen: set[str] = set()
    hits: list[dict] = []

    # --- Semantic search across rewritten query + all expanded alternatives ---
    base_vector = embed_query(rewritten, embed_model)
    for q in [rewritten] + alternatives:
        vector = embed_query(q, embed_model) if q != rewritten else base_vector
        for doc, meta, dist in _vector_search(vector, collection, top_k):
            if doc not in seen:
                seen.add(doc)
                hits.append({"document": doc, "metadata": meta, "distance": dist})

    # --- Keyword search for acronyms / specific terms not found semantically ---
    for term in key_terms(query):
        try:
            kw = collection.query(
                query_embeddings=[base_vector],
                where_document={"$contains": term},
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            for doc, meta, dist in zip(kw["documents"][0], kw["metadatas"][0], kw["distances"][0]):
                if doc not in seen:
                    seen.add(doc)
                    hits.append({"document": doc, "metadata": meta, "distance": dist})
        except Exception:
            pass

    hits.sort(key=lambda h: h["distance"])
    return hits[:top_k], rewritten


DENIAL = "Sorry, I don't have sufficient context about this, so cannot provide my response."


def filter_on_topic(query: str, hits: list[dict], chat_model: str) -> list[dict]:
    """Drop chunks whose subject matter doesn't match the query topic."""
    if not hits:
        return hits
    llm = ChatBedrockConverse(model_id=chat_model, region_name=config.BEDROCK_REGION)
    numbered = "\n\n".join(
        f"[{i + 1}] {h['document'][:400]}" for i, h in enumerate(hits)
    )
    prompt = (
        f"Question: {query}\n\n"
        f"Below are retrieved passages. Return ONLY the numbers of passages whose subject matter "
        f"directly addresses what the question is asking about. "
        f"A passage qualifies only if it is specifically about the same topic as the question — "
        f"not just about a related or similar topic. "
        f"Format: comma-separated numbers, e.g. 1,3. If none qualify, return: none\n\n"
        f"Passages:\n{numbered}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip().lower()
    if raw == "none":
        return []
    try:
        indices = [int(x.strip()) - 1 for x in raw.split(",") if x.strip().isdigit()]
        return [hits[i] for i in indices if 0 <= i < len(hits)]
    except Exception:
        return hits


def generate_answer(query: str, hits: list[dict], chat_model: str) -> str:
    relevant = [h for h in hits if h["distance"] <= config.DISTANCE_THRESHOLD]
    if not relevant:
        return DENIAL

    relevant = filter_on_topic(query, relevant, chat_model)
    if not relevant:
        return DENIAL

    context = "\n\n---\n\n".join(
        f"[Source: {h['metadata']['source']} | Section: {h['metadata'].get('section', 'N/A')} | chunk {h['metadata']['chunk_index']}]\n{h['document']}"
        for h in relevant
    )
    prompt = (
        "You are a precise assistant that only answers from the provided context.\n\n"
        "Instructions:\n"
        "1. Read the QUESTION carefully and identify its core intent.\n"
        "2. Check whether the CONTEXT directly and specifically addresses that intent.\n"
        "3. Before answering, check whether any chunks directly contradict each other on the exact topic of the QUESTION. "
        "A conflict is only relevant if BOTH contradicting chunks are specifically about what the QUESTION is asking. "
        "If the contradiction is between chunks on a different subject than the QUESTION, do NOT mention it at all — "
        "treat those chunks as if the contradiction does not exist. "
        "Only if a relevant conflict is found, flag it at the start of your answer including the file name and section of each conflicting source:\n"
        "   ⚠️ CONFLICT DETECTED: [describe the contradiction, naming the file and section of each disagreeing source]\n"
        "   Then present both sides and note the inconsistency.\n"
        "4. If no conflict exists and the context is relevant, answer concisely using ONLY information from the context.\n"
        f"5. If the context does not closely match the intent of the question, respond with exactly:\n"
        f'   "{DENIAL}"\n'
        "   Do NOT attempt to infer, guess, or use outside knowledge.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "ANSWER:"
    )
    llm = ChatBedrockConverse(model_id=chat_model, region_name=config.BEDROCK_REGION)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Search the local vector store and get an answer.")
    parser.add_argument("query",              help="The question or search query")
    parser.add_argument("--vectors-dir",      default=config.VECTORS_DIR,     help="Directory where ChromaDB is stored")
    parser.add_argument("--embed-model",      default=config.EMBED_MODEL,     help="model for query embedding")
    parser.add_argument("--chat-model",       default=config.CHAT_MODEL,      help="model for answer generation")
    parser.add_argument("--collection",       default=config.COLLECTION_NAME, help="ChromaDB collection name")
    parser.add_argument("--top-k",            default=10, type=int,            help="Number of chunks to retrieve")
    parser.add_argument("--no-generate",      action="store_true",            help="Only show retrieved chunks, skip answer generation")
    args = parser.parse_args()

    collection = get_collection(args.vectors_dir, args.collection)
    hits, rewritten = retrieve(args.query, collection, args.embed_model, args.chat_model, args.top_k)

    print(f"\n=== Rewritten query used for retrieval ===\n{rewritten}\n")
    print(f"=== Top {len(hits)} chunk(s) for: \"{args.query}\" ===\n")
    for i, hit in enumerate(hits, 1):
        source = hit["metadata"]["source"]
        chunk_idx = hit["metadata"]["chunk_index"]
        section = hit["metadata"].get("section", "")
        distance = hit["distance"]
        similarity = (1 - distance) * 100
        section_str = f"  section={section}" if section else ""
        print(f"[{i}] {source}  chunk={chunk_idx}{section_str}  distance={distance:.4f}  similarity={similarity:.1f}%")
        matched_sentence = hit["metadata"].get("sentence", "")
        display = matched_sentence if matched_sentence else hit["document"][:400]
        print(textwrap.indent(display, "    "))
        print()

    if not args.no_generate:
        print("=== Generated Answer ===\n")
        answer = generate_answer(args.query, hits, args.chat_model)
        print(answer)
        print()


if __name__ == "__main__":
    main()
