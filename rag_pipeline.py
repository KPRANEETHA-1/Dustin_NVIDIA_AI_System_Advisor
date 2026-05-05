"""RAG pipeline for NVIDIA AI Advisor: retrieval, grounding, generation, and output formatting."""

import os
import re
import json
import numpy as np
import faiss
import time

from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH   = os.path.join(SCRIPT_DIR, "faiss_index.index")
STORE_PATH   = os.path.join(SCRIPT_DIR, "chunk_store.json")
GEMINI_MODEL = "gemini-2.5-flash-lite"
TOP_K  = 3

# Minimum cosine score required before we call the LLM.
QUALITY_THRESHOLD = 0.25

OUTPUT_MODE = "user"

TOOL_KEYWORDS = {
    "nemo":     ["nemo", "fine-tune", "fine tune", "finetuning", "guardrails",
                 "evaluator", "data designer", "customizer", "synthetic data",
                 "workspace", "fileset"],
    "triton":   ["triton", "serving", "inference server", "dynamic batching",
                 "ensemble", "grpc", "kserve", "bls", "model repository",
                 "concurrent model"],
    "tensorrt": ["tensorrt", "trt", "trt-llm", "trtllm", "quantization",
                 "int8", "fp16", "fp8", "engine", "onnx", "optimization",
                 "build phase", "runtime phase", "precision"],
}

# NVIDIA topics outside the local NeMo/Triton/TensorRT knowledge base.
NVIDIA_ADJACENT_KEYWORDS = [
    "nvidia", "cuda", "gpu", "jetson", "drive", "omniverse", "isaac",
    "rapids", "cudf", "cuml", "merlin", "riva", "maxine", "broadcast",
    "rtx", "geforce", "quadro", "a100", "h100", "l40", "dgx", "hgx",
    "nim", "nvcf", "ai enterprise", "base command", "fleet command",
]

def is_conversational(query: str) -> bool:
    """Return True for greetings, thanks, goodbyes, and casual chat."""
    q = query.lower().strip()
    patterns = [
        "hi", "hey", "hello", "greetings", "howdy", "yo", "sup",
        "wassup", "what's up", "how are you", "how do you do",
        "do you copy", "you there", "you alive", "anyone home",
        "nice to meet", "pleasure", "good morning", "good evening",
        "thanks", "thank you", "appreciate it", "cheers",
        "bye", "goodbye", "see you", "later", "take care", "farewell",
        "that's all", "that's it", "done for now", "i'm done",
    ]
    return any(p in q for p in patterns)


def is_nvidia_adjacent(query: str) -> bool:
    """Return True for NVIDIA queries outside the three indexed tools."""
    q = query.lower()
    all_tool_kws = [kw for kws in TOOL_KEYWORDS.values() for kw in kws]
    if any(kw in q for kw in all_tool_kws):
        return False
    return any(kw in q for kw in NVIDIA_ADJACENT_KEYWORDS)


def handle_conversational(query: str, mode: str, client) -> "PipelineOutput":
    """Handle conversational prompts without retrieval."""
    dustin_persona = """You are Dustin Henderson from Stranger Things.
You are enthusiastic, warm, and genuinely excited about technology.
You are acting as an AI assistant called the NVIDIA AI System Advisor â€”
you help users understand NeMo Platform, Triton Inference Server, and TensorRT.

The user is greeting you, saying goodbye, thanking you, or just chatting.
Respond naturally as Dustin would â€” warm, a little excitable, genuine.
Keep it brief (2-4 sentences max).
If greeting: respond warmly and mention you can help with the NVIDIA AI stack.
If goodbye/thanks: respond warmly and sign off as Dustin would naturally.
Do NOT lecture. Do NOT list features. Just be natural."""

    msg = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=query,
        config=types.GenerateContentConfig(
            system_instruction=dustin_persona,
        )
    ).text.strip()

    dummy = RetrievalResult(
        chunks=[], top_score=0.0,
        detected_tool=None, passed_check=False, query=query
    )
    return PipelineOutput(
        query=query, raw_answer=msg, formatted_answer=msg,
        final_answer=msg, sources=[], retrieval=dummy, mode=mode,
    )


def handle_nvidia_adjacent(query: str, mode: str, client) -> "PipelineOutput":
    """Redirect NVIDIA-adjacent questions outside the local knowledge base."""
    dustin_redirect = """You are Dustin Henderson from Stranger Things,
acting as the NVIDIA AI System Advisor. Your specific knowledge base covers:
NeMo Platform, Triton Inference Server, and TensorRT / TensorRT-LLM.

The user asked an NVIDIA-related question outside your specific knowledge base.
Respond as Dustin â€” acknowledge it is a great question, be honest that it is
outside your specific docs, and enthusiastically redirect them to NVIDIA's
official documentation. Include this link naturally: https://docs.nvidia.com

3-4 sentences max. Warm and natural. No bullet points. No robotic phrasing."""

    msg = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=query,
        config=types.GenerateContentConfig(
            system_instruction=dustin_redirect,
        )
    ).text.strip()

    dummy = RetrievalResult(
        chunks=[], top_score=0.0,
        detected_tool=None, passed_check=False, query=query
    )
    return PipelineOutput(
        query=query, raw_answer=msg, formatted_answer=msg,
        final_answer=msg, sources=[], retrieval=dummy, mode=mode,
    )


@dataclass
class RetrievalResult:
    """
    Output of the retrieval stage. Passed to the generation stage.
    Keeping this as a dataclass means every downstream function gets a
    predictable, typed object â€” no dict key typos, no missing fields.
    """
    chunks:        list[dict]    # retrieved chunk objects with similarity scores
    top_score:     float         # highest similarity score among retrieved chunks
    detected_tool: str | None    # which tool the query was routed to (or None)
    passed_check:  bool          # did retrieval quality check pass?
    query:         str           # the original user query


@dataclass
class PipelineOutput:
    """
    Final output of the full pipeline. Passed to the display layer.
    Separates raw answer, formatted answer, and toned answer so we can
    show different amounts of detail in debug vs user mode.
    """
    query:            str
    raw_answer:       str
    formatted_answer: str
    final_answer:     str
    sources:          list[dict]   # list of {section, source, doc_link} dicts
    retrieval:        RetrievalResult
    mode:             str          # "user" or "debug"


# Do NOT modify these functions. They are stable and tested.

def load_artifacts() -> tuple:
    """Loads FAISS index, chunk store, and embedding model at startup."""
    print("Loading RAG artifacts...")

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"faiss_index.index not found.\nRun embedder.py first."
        )
    index = faiss.read_index(INDEX_PATH)
    print(f"  âœ… FAISS index     ({index.ntotal} vectors)")

    if not os.path.exists(STORE_PATH):
        raise FileNotFoundError(
            f"chunk_store.json not found.\nRun embedder.py first."
        )
    with open(STORE_PATH, "r", encoding="utf-8") as f:
        chunk_store = json.load(f)

    chunks   = chunk_store["chunks"]
    metadata = chunk_store["metadata"]
    print(f"  âœ… Chunk store     ({len(chunks)} chunks)")

    model_name  = metadata["model_name"]
    embed_model = SentenceTransformer(model_name, trust_remote_code=True)
    print(f"  âœ… Embedding model ({model_name})")

    return index, chunks, embed_model, metadata


def detect_source(query: str) -> str | None:
    """Detects which NVIDIA tool the query is about from keyword matching."""
    query_lower = query.lower()
    scores = {}
    for source, keywords in TOOL_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in query_lower)
        if hits > 0:
            scores[source] = hits
    return max(scores, key=scores.get) if scores else None


def retrieve_chunks(
    query: str,
    index: faiss.Index,
    chunks: list[dict],
    embed_model: SentenceTransformer,
    top_k: int = TOP_K
) -> RetrievalResult:
    """
    Embeds the query, routes to the right source pool, searches FAISS.
    Returns a RetrievalResult (unchanged logic from v1, wrapped in dataclass).
    """
    query_vec = embed_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    detected = detect_source(query)

    if detected:
        priority_indices = [i for i, c in enumerate(chunks) if c["source"] == detected]
        fallback_indices = [i for i, c in enumerate(chunks) if c["source"] != detected]
    else:
        priority_indices = list(range(len(chunks)))
        fallback_indices = []

    def search_subset(candidate_indices: list[int], k: int) -> list[dict]:
        if not candidate_indices:
            return []
        k_fetch = min(len(candidate_indices), top_k * 5)
        distances, indices = index.search(query_vec, k=k_fetch)
        candidate_set = set(candidate_indices)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx in candidate_set:
                chunk = dict(chunks[idx])
                chunk["similarity_score"] = float(dist)
                results.append(chunk)
                if len(results) >= k:
                    break
        return results

    results = search_subset(priority_indices, top_k)
    if len(results) < top_k and fallback_indices:
        needed = top_k - len(results)
        results.extend(search_subset(fallback_indices, needed))

    results = results[:top_k]
    top_score = results[0]["similarity_score"] if results else 0.0

    return RetrievalResult(
        chunks=results,
        top_score=top_score,
        detected_tool=detected,
        passed_check=False,   # set by quality_check() below
        query=query,
    )


# All modifications go below this line.



def quality_check(retrieval: RetrievalResult) -> RetrievalResult:
    """
    Gates the LLM call. If retrieved chunks aren't relevant enough,
    we skip Claude entirely and return a "not found" signal.

    WHY THIS MATTERS:
    Without this gate, a query like "what is the best pizza in Rome?"
    would still retrieve the 3 least-irrelevant chunks from our NVIDIA docs,
    pass them to Claude as context, and generate a confidently wrong answer
    that mixes pizza with TensorRT. That's the classic RAG hallucination pattern.

    With the gate: low score â†’ passed_check=False â†’ pipeline returns a
    "not in my knowledge base" response immediately. No LLM call, no cost,
    no hallucination.

    SCORE INTERPRETATION (MiniLM cosine similarity):
      > 0.50  = strong match â€” chunk is clearly relevant
      0.35-0.50 = good match â€” chunk is relevant
      0.25-0.35 = weak match â€” chunk is marginally relevant
      < 0.25  = no match â€” query is outside the knowledge base
    """
    retrieval.passed_check = retrieval.top_score >= QUALITY_THRESHOLD
    return retrieval



def build_grounded_prompt(query: str, retrieval: RetrievalResult) -> tuple[str, str]:
    """
    Builds the system prompt with STRONG grounding rules.

    V1 vs V2 system prompt difference:
    V1: "Base your answer primarily on the provided context"
        (soft â€” Claude could still supplement from training)

    V2: Explicit numbered rules with hard constraints:
        - Must cite which context section you used
        - Must say "not in provided documentation" if context is insufficient
        - Forbidden from using external knowledge as primary source
        - Forbidden from speculating about version numbers, pricing, roadmaps

    WHY STRICT RULES MATTER:
    LLMs have a strong prior toward being helpful â€” they'll answer even when
    they shouldn't. Without hard rules, Claude might blend retrieved context
    with training knowledge in a way that's undetectable. For a technical
    advisor system, a confidently wrong answer is worse than "I don't know."
    """
    context_blocks = []
    for i, chunk in enumerate(retrieval.chunks, 1):
        block = (
            f"[CONTEXT {i}]\n"
            f"Source  : {chunk['source'].upper()}\n"
            f"Section : {chunk['section']}\n"
            f"Link    : {chunk['doc_link']}\n"
            f"Content :\n{chunk['content']}"
        )
        context_blocks.append(block)

    context_text = "\n\n".join(context_blocks)

    system_prompt = f"""You are the NVIDIA AI System Advisor â€” a precise technical assistant \
for the NVIDIA AI ecosystem (NeMo Platform, Triton Inference Server, TensorRT / TensorRT-LLM).

â”â”â” GROUNDING RULES (non-negotiable) â”â”â”

RULE 1 â€” CONTEXT FIRST
Your answer must be grounded in the provided CONTEXT SECTIONS below.
Use your general knowledge only to explain terminology â€” never as the primary source.

RULE 2 â€” CITE YOUR SOURCES
End every answer with a "Sources Used:" section listing which [CONTEXT N] blocks
you drew from, with their section name and doc link.

RULE 3 â€” ADMIT GAPS HONESTLY
If the provided context does not fully answer the question, say:
"The provided documentation does not cover [topic]. For complete information, see [doc_link]."
Do NOT speculate or fill gaps with assumed knowledge.

RULE 4 â€” FORBIDDEN CONTENT
Never state specific version numbers, pricing, release dates, or roadmap items
unless they appear verbatim in the provided context sections.

RULE 5 â€” TOOL RECOMMENDATIONS
If the question asks which tool to use, give a clear recommendation with reasoning
drawn ONLY from the context. Do not recommend tools not mentioned in the context.

â”â”â” CONTEXT SECTIONS â”â”â”

{context_text}

â”â”â” ANSWER FORMAT â”â”â”

Structure your response as:
ANSWER:
[your answer here â€” concise, technically precise, grounded in context]

SOURCES USED:
- [CONTEXT N] â€” Section name â€” doc link
(list every context section you referenced)"""

    user_message = f"Question: {query}"
    return system_prompt, user_message



def format_answer(raw_answer: str, retrieval: RetrievalResult) -> tuple[str, list[dict]]:
    """
    Parses Claude's raw output into a clean answer body + structured sources list.

    WHY A SEPARATE FORMATTING LAYER?
    Claude's raw output follows our format template but may have:
    - Extra whitespace or markdown artifacts
    - Sources mixed into the answer body
    - Inconsistent line breaks

    By parsing here, we ensure the downstream Dustin tone layer only sees
    the clean answer text â€” not the sources block â€” and we produce a
    machine-readable sources list that the display layer can render
    consistently regardless of how Claude formatted it.

    PARSING STRATEGY:
    Split on "SOURCES USED:" to separate answer body from sources block.
    Then clean the answer body and extract individual source lines.
    Falls back gracefully if Claude didn't follow the format exactly.
    """

    sources_marker = "SOURCES USED:"
    answer_marker  = "ANSWER:"

    if sources_marker in raw_answer:
        parts        = raw_answer.split(sources_marker, 1)
        answer_body  = parts[0].strip()
        sources_text = parts[1].strip()
    else:
        # Claude didn't include sources section â€” use full response as answer
        answer_body  = raw_answer.strip()
        sources_text = ""

    # Strip the "ANSWER:" label if present
    if answer_marker in answer_body:
        answer_body = answer_body.split(answer_marker, 1)[1].strip()

    # We use the retrieval result directly (ground truth) rather than parsing
    # Claude's sources text, which could be incomplete or malformatted.
    sources = []
    for chunk in retrieval.chunks:
        sources.append({
            "section":  chunk["section"],
            "source":   chunk["source"].upper(),
            "doc_link": chunk["doc_link"],
            "score":    chunk.get("similarity_score", 0.0),
        })

    return answer_body.strip(), sources



def apply_dustin_tone(formatted_answer: str, client=None) -> str:
    """
    Applies Dustin Henderson's personality to the formatted answer.

    WHAT THIS LAYER DOES:
    Sends the clean answer to Claude with a tight persona prompt that
    rewrites the tone while preserving every factual detail.

    DUSTIN TONE RULES (enforced in the prompt):
    1. Do NOT change factual meaning â€” same information, different voice
    2. Do NOT add new information â€” zero hallucination risk
    3. Keep explanation clear and structured â€” no sacrifice of clarity
    4. Add light conversational phrasing â€” 1-2 sentences max per section
    5. Use analogies sparingly â€” only when they genuinely help
    6. No excessive slang, emojis, or over-casual language
    7. 80% professional, 20% personality

    WHY A SEPARATE API CALL FOR TONE?
    Doing tone transformation in the same call as retrieval + grounding
    creates conflicting objectives â€” the grounding rules say "be precise"
    while the tone rules say "be conversational". Separating them lets each
    Claude call do one job perfectly. The cost is one extra API call, but
    the quality difference is significant.

    SAFETY: The tone prompt explicitly forbids adding content. The only
    change allowed is phrasing. This makes it impossible for the tone layer
    to introduce hallucinations.
    """

    tone_system = """You are applying a light personality layer to a technical answer.

The character is Dustin Henderson from Stranger Things â€” enthusiastic, smart, 
uses clear analogies when helpful, genuinely excited about technology, but 
fundamentally a precise and reliable explainer.

STRICT RULES â€” violating these is not allowed:
1. DO NOT change any factual content â€” same facts, same structure
2. DO NOT add any new information â€” zero additions
3. DO NOT remove any technical detail â€” keep everything
4. ADD only: a warm opening phrase, occasional "here's the thing" transitions,
   one light analogy per section if it genuinely helps understanding
5. NO excessive slang, NO emojis, NO "dude/bro/man" more than once total
6. 80% of words must be the original professional content
7. Keep all section headers and the answer structure intact
8. Sources section: leave completely unchanged â€” do not touch it

Transform the tone, not the content."""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"Apply Dustin tone to this answer:\n\n{formatted_answer}",
        config=types.GenerateContentConfig(
            system_instruction=tone_system,
        )
    )
    return response.text.strip()



def print_output(output: PipelineOutput) -> None:
    """
    Renders the final pipeline output.

    USER MODE (default):
      Shows only the final toned answer + sources.
      Clean, no diagnostic noise. What an end user sees.

    DEBUG MODE:
      Shows everything: similarity scores, confidence bars, chunk previews,
      raw Claude answer before formatting, formatted answer before tone,
      and the final toned answer. What a developer sees when tuning the system.

    HOW TO SWITCH MODES:
      Set OUTPUT_MODE = "debug" at the top of this file
      Or pass mode="debug" when constructing PipelineOutput
    """
    mode = output.mode
    print("\n" + "â•" * 64)
    print(f"  {'ðŸ”§ DEBUG MODE' if mode == 'debug' else 'ðŸ’¬ NVIDIA AI Advisor'}")
    print(f"  Q: {output.query}")
    print("â•" * 64)

    if mode == "debug":
        r = output.retrieval
        print(f"\n  RETRIEVAL DIAGNOSTICS")
        print(f"  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
        print(f"  Detected tool  : {r.detected_tool or 'None (general query)'}")
        print(f"  Top score      : {r.top_score:.4f}  "
              f"{'âœ… passed' if r.passed_check else 'âŒ failed'} "
              f"(threshold: {QUALITY_THRESHOLD})")

        print(f"\n  Retrieved chunks:")
        for i, chunk in enumerate(r.chunks, 1):
            score   = chunk.get("similarity_score", 0)
            bar_len = min(int(score * 20), 20)
            bar     = "â–ˆ" * bar_len + "â–‘" * (20 - bar_len)
            print(f"\n  #{i}  [{chunk['chunk_id']}]  score={score:.4f}  {bar}")
            print(f"       Source  : {chunk['source'].upper()}")
            print(f"       Section : {chunk['section']}")
            print(f"       Preview : {chunk['content'][:120].replace(chr(10), ' ')}...")

        print(f"\n  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
        print(f"  RAW ANSWER (from Claude, before formatting):\n")
        for line in output.raw_answer.split("\n"):
            print(f"    {line}")

        print(f"\n  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
        print(f"  FORMATTED ANSWER (after format_answer(), before tone):\n")
        for line in output.formatted_answer.split("\n"):
            print(f"    {line}")

        print(f"\n  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
        print(f"  FINAL ANSWER (after apply_dustin_tone()):\n")

    print()
    for line in output.final_answer.split("\n"):
        print(f"  {line}")

    print(f"\n  {'â”€' * 60}")
    print(f"  SOURCES")
    for src in output.sources:
        score_indicator = f"(score: {src['score']:.2f})" if mode == "debug" else ""
        print(f"  â€¢ [{src['source']}] {src['section']} {score_indicator}")
        print(f"    {src['doc_link']}")

    print("\n" + "â•" * 64)


# NOT-FOUND RESPONSE â€” returned when quality check fails

def not_found_output(query: str, retrieval: RetrievalResult, mode: str) -> PipelineOutput:
    """
    Returns a graceful "not found" response when retrieval quality check fails.
    No LLM call made. No hallucination risk.
    """
    msg = (
        "That question doesn't appear to be covered in the NVIDIA AI documentation "
        "I have access to (NeMo, Triton, TensorRT). "
        f"The best match I found had a similarity score of {retrieval.top_score:.2f}, "
        f"which is below my confidence threshold of {QUALITY_THRESHOLD}.\n\n"
        "Try rephrasing with specific tool names like 'NeMo', 'Triton', or 'TensorRT', "
        "or check the official docs directly:\n"
        "â€¢ NeMo    : https://docs.nvidia.com/nemo-framework/user-guide/latest/index.html\n"
        "â€¢ Triton  : https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html\n"
        "â€¢ TensorRT: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html"
    )
    return PipelineOutput(
        query=query,
        raw_answer=msg,
        formatted_answer=msg,
        final_answer=msg,
        sources=[],
        retrieval=retrieval,
        mode=mode,
    )


# MASTER PIPELINE FUNCTION

def run_pipeline(
    query: str,
    index: faiss.Index,
    chunks: list[dict],
    embed_model: SentenceTransformer,
    client,
    mode: str = OUTPUT_MODE,
) -> PipelineOutput:
    """Run retrieval, grounding, generation, formatting, and tone."""

    if is_conversational(query):
        return handle_conversational(query, mode, client)

    if is_nvidia_adjacent(query):
        return handle_nvidia_adjacent(query, mode, client)

    retrieval = retrieve_chunks(query, index, chunks, embed_model)

    retrieval = quality_check(retrieval)
    if not retrieval.passed_check:
        return not_found_output(query, retrieval, mode)

    system_prompt, user_message = build_grounded_prompt(query, retrieval)

    raw_answer = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
        )
    ).text

    formatted_answer, sources = format_answer(raw_answer, retrieval)

    final_answer = apply_dustin_tone(formatted_answer, client)

    return PipelineOutput(
        query=query,
        raw_answer=raw_answer,
        formatted_answer=formatted_answer,
        final_answer=final_answer,
        sources=sources,
        retrieval=retrieval,
        mode=mode,
    )



def main():
    print("=" * 64)
    print("  NVIDIA AI System Advisor v2")
    print("  MiniLM + FAISS + Gemini 2.5 Flash Lite + Dustin")
    print(f"  Mode: {OUTPUT_MODE.upper()}")
    print("=" * 64)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "\nGEMINI_API_KEY not set.\n"
            "Add to .env file: GEMINI_API_KEY=AIza...\n"
            "Get your free key at: https://aistudio.google.com\n"
        )

    # Reuse one Gemini client for all generation calls.
    client = genai.Client(api_key=api_key)

    # Load all artifacts at startup
    index, chunks, embed_model, _ = load_artifacts()
    print(f"\n  âœ… Gemini ready  ({GEMINI_MODEL})")

    # Demo queries
    demo_queries = [
        "What is NeMo Platform and when should I use it?",
        "How does Triton handle dynamic batching?",
        "What is the difference between NeMo, Triton, and TensorRT?",
        "How do I make a pizza?",   # â† intentional out-of-domain query to test quality gate
    ]

    print("\n" + "=" * 64)
    print("  DEMO QUERIES")
    print("=" * 64)
    
    for query in demo_queries:
        print(f"\n  Running: \"{query}\"")
        output = run_pipeline(query, index, chunks, embed_model, client, mode=OUTPUT_MODE)
        print_output(output)
        time.sleep(8) 

    # Interactive loop
    print("\n" + "=" * 64)
    print(f"  INTERACTIVE MODE  |  mode={OUTPUT_MODE}  |  type 'exit' to quit")
    print(f"  Tip: set OUTPUT_MODE='debug' at top of file for full diagnostics")
    print("=" * 64)

    while True:
        print()
        query = input("  Your question: ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("\n  Later! â€” Dustin")
            break

        output = run_pipeline(query, index, chunks, embed_model, client, mode=OUTPUT_MODE)
        print_output(output)


if __name__ == "__main__":
    main()





