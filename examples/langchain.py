#!/usr/bin/env python3
"""
langchain.py — ferrumox + LangChain integration examples.

Demonstrates:
  - Basic chat with ChatOpenAI
  - Streaming tokens
  - RAG pipeline with embeddings
  - Structured output with JSON mode

Prerequisites:
  pip install langchain langchain-openai

Usage:
  # Start ferrumox first
  fox serve --model-path ~/.cache/ferrumox/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf

  # Run examples
  python examples/langchain.py
"""

import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# ── Configuration ─────────────────────────────────────────────────────────────

FERRUMOX_URL = os.getenv("FERRUMOX_URL", "http://localhost:8080/v1")
MODEL = os.getenv("FOX_MODEL", "default")

# ferrumox accepts any non-empty string as the API key
API_KEY = "fox"


# ── Helper: build LLM client ──────────────────────────────────────────────────

def make_llm(streaming: bool = False, **kwargs) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=FERRUMOX_URL,
        api_key=API_KEY,
        model=MODEL,
        streaming=streaming,
        **kwargs,
    )


def make_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        base_url=FERRUMOX_URL,
        api_key=API_KEY,
        model=MODEL,
    )


# ── Example 1: Basic chat ─────────────────────────────────────────────────────

def example_basic_chat():
    print("=== Example 1: Basic chat ===")
    llm = make_llm(temperature=0.7, max_tokens=128)

    messages = [
        SystemMessage(content="You are a concise assistant. Answer in one sentence."),
        HumanMessage(content="What makes Rust different from C++?"),
    ]

    response = llm.invoke(messages)
    print(f"Response: {response.content}")
    print()


# ── Example 2: Streaming ──────────────────────────────────────────────────────

def example_streaming():
    print("=== Example 2: Streaming ===")
    llm = make_llm(streaming=True, max_tokens=64)

    print("Streaming: ", end="", flush=True)
    for chunk in llm.stream("List 3 benefits of Rust in one line each."):
        print(chunk.content, end="", flush=True)
    print("\n")


# ── Example 3: Prompt template + chain ───────────────────────────────────────

def example_chain():
    print("=== Example 3: Prompt template + LLMChain ===")
    llm = make_llm(max_tokens=128)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful programming tutor."),
        ("human", "Explain {concept} in simple terms for a {level} developer."),
    ])

    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(concept="ownership and borrowing", level="beginner")
    print(f"Result: {result}")
    print()


# ── Example 4: Embeddings ─────────────────────────────────────────────────────

def example_embeddings():
    print("=== Example 4: Embeddings ===")
    embeddings = make_embeddings()

    texts = [
        "Rust prevents memory safety bugs at compile time.",
        "Python is easy to learn and great for scripting.",
        "C++ gives you full control over memory management.",
    ]

    vectors = embeddings.embed_documents(texts)
    print(f"Embedded {len(vectors)} documents, dimension = {len(vectors[0])}")

    # Simple cosine similarity to find the most similar pair
    import math

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b + 1e-9)

    query = embeddings.embed_query("memory safety in systems languages")
    sims = [(cosine(query, v), t) for v, t in zip(vectors, texts)]
    sims.sort(reverse=True)
    print("Most similar to 'memory safety in systems languages':")
    for score, text in sims:
        print(f"  {score:.3f}  {text}")
    print()


# ── Example 5: Structured output (JSON mode) ──────────────────────────────────

def example_json_mode():
    print("=== Example 5: Structured output (JSON mode) ===")
    llm = make_llm(
        max_tokens=128,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    response = llm.invoke(
        "Return a JSON object with: name (string), year_created (number), "
        "paradigms (array of strings) for the Rust programming language."
    )
    import json
    parsed = json.loads(response.content)
    print(f"Parsed JSON: {json.dumps(parsed, indent=2)}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"ferrumox LangChain examples")
    print(f"  URL   : {FERRUMOX_URL}")
    print(f"  Model : {MODEL}")
    print()

    example_basic_chat()
    example_streaming()
    example_chain()
    example_embeddings()
    example_json_mode()

    print("All examples completed.")
