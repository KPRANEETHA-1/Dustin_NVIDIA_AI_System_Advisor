"""
NVIDIA AI Advisor — RAG Pipeline v2
=====================================
Upgrades over v1 (retrieval logic unchanged):

  LAYER 1 — GROUNDING RULES
    System prompt now has strict anti-hallucination rules:
    Claude must answer only from context, never invent, must signal
    when context is insufficient.

  LAYER 2 — RETRIEVAL QUALITY CHECK
    Before calling the LLM, we check if retrieved chunks are actually
    relevant. If the best similarity score is too low, we skip the LLM
    call entirely and return a "not found" response. No wasted API calls
    on irrelevant retrievals.

  LAYER 3 — ANSWER FORMATTING
    raw_answer from Claude → format_answer() → clean structured output
    Separates answer body from sources. Consistent every time.

  LAYER 4 — DUSTIN TONE
    format_answer() → apply_dustin_tone() → final human-friendly output
    Dustin Henderson (Stranger Things) personality layer: 80% professional,
    20% conversational warmth. Does NOT change facts. Does NOT add info.
    Is a pure UX transformation — same answer, more approachable delivery.

  LAYER 5 — OUTPUT MODES
    DEBUG mode: shows similarity scores, chunk previews, raw answer
    USER mode : shows only final answer + sources (default)

NEW PIPELINE FLOW:
  User question
      ↓  embed (MiniLM)            ← UNCHANGED
  Query vector
      ↓  FAISS search + routing    ← UNCHANGED
  Retrieved chunks
      ↓  quality_check()           ← NEW: gate before LLM
  Augmented prompt (grounding rules) ← NEW: stronger system prompt
      ↓  Claude API
  raw_answer
      ↓  format_answer()           ← NEW: structure extraction
  formatted_answer
      ↓  apply_dustin_tone()       ← NEW: personality layer
  final_output
      ↓  print_output(mode)        ← NEW: debug vs user mode

DO NOT MODIFY: load_artifacts(), retrieve_chunks(), detect_source()
MODIFY FREELY:  everything below the "GENERATION + UX LAYERS" marker
"""

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

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH   = os.path.join(SCRIPT_DIR, "faiss_index.index")
STORE_PATH   = os.path.join(SCRIPT_DIR, "chunk_store.json")
# gemini-2.5-flash-lite: fast, free tier available, great for RAG Q&A.
# Free tier: 15 requests/min, 1 million tokens/day — more than enough.
GEMINI_MODEL = "gemini-2.5-flash-lite"
TOP_K        = 3

# LAYER 2 — Retrieval quality threshold
# Cosine similarity below this = retrieved chunks are not relevant enough.
# 0.25 is calibrated for MiniLM on technical domain queries.
# Raise to 0.35 for stricter filtering. Lower to 0.15 for more permissive.
QUALITY_THRESHOLD = 0.25

# Output mode: "user" (default, clean output) or "debug" (full diagnostics)
# Change to "debug" when developing / testing retrieval quality
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

# NVIDIA-adjacent keywords — user is asking about NVIDIA broadly
# but NOT about the 3 tools we have docs for
NVIDIA_ADJACENT_KEYWORDS = [
    "nvidia", "cuda", "gpu", "jetson", "drive", "omniverse", "isaac",
    "rapids", "cudf", "cuml", "merlin", "riva", "maxine", "broadcast",
    "rtx", "geforce", "quadro", "a100", "h100", "l40", "dgx", "hgx",
    "nim", "nvcf", "ai enterprise", "base command", "fleet command",
]


# ─────────────────────────────────────────────────────────────────────────────
# CONVERSATIONAL + NVIDIA-ADJACENT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def is_conversational(query: str) -> bool:
    """
    Detects greetings, goodbyes, thanks, casual chat.
    These bypass quality gate — LLM handles them naturally as Dustin.
    No hardcoded responses. The model reads the intent and replies freely.
    """
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
    """
    Detects NVIDIA-related queries outside our 3-tool knowledge base.
    Only triggers if the query does NOT already match NeMo/Triton/TensorRT
    keywords — those go through the normal retrieval pipeline.
    """
    q = query.lower()
    all_tool_kws = [kw for kws in TOOL_KEYWORDS.values() for kw in kws]
    if any(kw in q for kw in all_tool_kws):
        return False   # already handled by main pipeline
    return any(kw in q for kw in NVIDIA_ADJACENT_KEYWORDS)


def handle_conversational(query: str, mode: str, client) -> "PipelineOutput":
    """
    Routes conversational queries to Gemini with a pure Dustin persona.
    No retrieval context. Model responds naturally to the greeting/goodbye.
    Lightweight prompt — minimal quota usage.
    """
    dustin_persona = """You are Dustin Henderson from Stranger Things.
You are enthusiastic, warm, and genuinely excited about technology.
You are acting as an AI assistant called the NVIDIA AI System Advisor —
you help users understand NeMo Platform, Triton Inference Server, and TensorRT.

The user is greeting you, saying goodbye, thanking you, or just chatting.
Respond naturally as Dustin would — warm, a little excitable, genuine.
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
    """
    Routes NVIDIA-adjacent queries to Gemini with a redirect prompt.
    Dustin acknowledges the question, is honest about his scope,
    and naturally points to NVIDIA official docs with the link.
    """
    dustin_redirect = """You are Dustin Henderson from Stranger Things,
acting as the NVIDIA AI System Advisor. Your specific knowledge base covers:
NeMo Platform, Triton Inference Server, and TensorRT / TensorRT-LLM.

The user asked an NVIDIA-related question outside your specific knowledge base.
Respond as Dustin — acknowledge it is a great question, be honest that it is
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


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# Two clean containers that flow through the pipeline stages.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """
    Output of the retrieval stage. Passed to the generation stage.
    Keeping this as a dataclass means every downstream function gets a
    predictable, typed object — no dict key typos, no missing fields.
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
    raw_answer:       str          # exactly what Claude returned
    formatted_answer: str          # after format_answer() — clean structure
    final_answer:     str          # after apply_dustin_tone() — human-friendly
    sources:          list[dict]   # list of {section, source, doc_link} dicts
    retrieval:        RetrievalResult
    mode:             str          # "user" or "debug"


# ═══════════════════════════════════════════════════════════════════════════════
# ── UNCHANGED: RETRIEVAL LAYER ────────────────────────────────────────────────
# Do NOT modify these functions. They are stable and tested.
# ═══════════════════════════════════════════════════════════════════════════════

def load_artifacts() -> tuple:
    """Loads FAISS index, chunk store, and embedding model at startup."""
    print("Loading RAG artifacts...")

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"faiss_index.index not found.\nRun embedder.py first."
        )
    index = faiss.read_index(INDEX_PATH)
    print(f"  ✅ FAISS index     ({index.ntotal} vectors)")

    if not os.path.exists(STORE_PATH):
        raise FileNotFoundError(
            f"chunk_store.json not found.\nRun embedder.py first."
        )
    with open(STORE_PATH, "r", encoding="utf-8") as f:
        chunk_store = json.load(f)

    chunks   = chunk_store["chunks"]
    metadata = chunk_store["metadata"]
    print(f"  ✅ Chunk store     ({len(chunks)} chunks)")

    model_name  = metadata["model_name"]
    embed_model = SentenceTransformer(model_name, trust_remote_code=True)
    print(f"  ✅ Embedding model ({model_name})")

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


# ═══════════════════════════════════════════════════════════════════════════════
# ── NEW: GENERATION + UX LAYERS ───────────────────────────────────────────────
# All modifications go below this line.
# ═══════════════════════════════════════════════════════════════════════════════


# ── LAYER 2: RETRIEVAL QUALITY CHECK ─────────────────────────────────────────

def quality_check(retrieval: RetrievalResult) -> RetrievalResult:
    """
    Gates the LLM call. If retrieved chunks aren't relevant enough,
    we skip Claude entirely and return a "not found" signal.

    WHY THIS MATTERS:
    Without this gate, a query like "what is the best pizza in Rome?"
    would still retrieve the 3 least-irrelevant chunks from our NVIDIA docs,
    pass them to Claude as context, and generate a confidently wrong answer
    that mixes pizza with TensorRT. That's the classic RAG hallucination pattern.

    With the gate: low score → passed_check=False → pipeline returns a
    "not in my knowledge base" response immediately. No LLM call, no cost,
    no hallucination.

    SCORE INTERPRETATION (MiniLM cosine similarity):
      > 0.50  = strong match — chunk is clearly relevant
      0.35-0.50 = good match — chunk is relevant
      0.25-0.35 = weak match — chunk is marginally relevant
      < 0.25  = no match — query is outside the knowledge base
    """
    retrieval.passed_check = retrieval.top_score >= QUALITY_THRESHOLD
    return retrieval


# ── LAYER 1: GROUNDING RULES IN SYSTEM PROMPT ────────────────────────────────

def build_grounded_prompt(query: str, retrieval: RetrievalResult) -> tuple[str, str]:
    """
    Builds the system prompt with STRONG grounding rules.

    V1 vs V2 system prompt difference:
    V1: "Base your answer primarily on the provided context"
        (soft — Claude could still supplement from training)

    V2: Explicit numbered rules with hard constraints:
        - Must cite which context section you used
        - Must say "not in provided documentation" if context is insufficient
        - Forbidden from using external knowledge as primary source
        - Forbidden from speculating about version numbers, pricing, roadmaps

    WHY STRICT RULES MATTER:
    LLMs have a strong prior toward being helpful — they'll answer even when
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

    system_prompt = f"""You are the NVIDIA AI System Advisor — a precise technical assistant \
for the NVIDIA AI ecosystem (NeMo Platform, Triton Inference Server, TensorRT / TensorRT-LLM).

━━━ GROUNDING RULES (non-negotiable) ━━━

RULE 1 — CONTEXT FIRST
Your answer must be grounded in the provided CONTEXT SECTIONS below.
Use your general knowledge only to explain terminology — never as the primary source.

RULE 2 — CITE YOUR SOURCES
End every answer with a "Sources Used:" section listing which [CONTEXT N] blocks
you drew from, with their section name and doc link.

RULE 3 — ADMIT GAPS HONESTLY
If the provided context does not fully answer the question, say:
"The provided documentation does not cover [topic]. For complete information, see [doc_link]."
Do NOT speculate or fill gaps with assumed knowledge.

RULE 4 — FORBIDDEN CONTENT
Never state specific version numbers, pricing, release dates, or roadmap items
unless they appear verbatim in the provided context sections.

RULE 5 — TOOL RECOMMENDATIONS
If the question asks which tool to use, give a clear recommendation with reasoning
drawn ONLY from the context. Do not recommend tools not mentioned in the context.

━━━ CONTEXT SECTIONS ━━━

{context_text}

━━━ ANSWER FORMAT ━━━

Structure your response as:
ANSWER:
[your answer here — concise, technically precise, grounded in context]

SOURCES USED:
- [CONTEXT N] — Section name — doc link
(list every context section you referenced)"""

    user_message = f"Question: {query}"
    return system_prompt, user_message


# ── LAYER 3: ANSWER FORMATTING ────────────────────────────────────────────────

def format_answer(raw_answer: str, retrieval: RetrievalResult) -> tuple[str, list[dict]]:
    """
    Parses Claude's raw output into a clean answer body + structured sources list.

    WHY A SEPARATE FORMATTING LAYER?
    Claude's raw output follows our format template but may have:
    - Extra whitespace or markdown artifacts
    - Sources mixed into the answer body
    - Inconsistent line breaks

    By parsing here, we ensure the downstream Dustin tone layer only sees
    the clean answer text — not the sources block — and we produce a
    machine-readable sources list that the display layer can render
    consistently regardless of how Claude formatted it.

    PARSING STRATEGY:
    Split on "SOURCES USED:" to separate answer body from sources block.
    Then clean the answer body and extract individual source lines.
    Falls back gracefully if Claude didn't follow the format exactly.
    """

    # ── Split answer body from sources block ─────────────────────────────────
    sources_marker = "SOURCES USED:"
    answer_marker  = "ANSWER:"

    if sources_marker in raw_answer:
        parts        = raw_answer.split(sources_marker, 1)
        answer_body  = parts[0].strip()
        sources_text = parts[1].strip()
    else:
        # Claude didn't include sources section — use full response as answer
        answer_body  = raw_answer.strip()
        sources_text = ""

    # Strip the "ANSWER:" label if present
    if answer_marker in answer_body:
        answer_body = answer_body.split(answer_marker, 1)[1].strip()

    # ── Build structured sources list from retrieval result ───────────────────
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


# ── LAYER 4: DUSTIN TONE ──────────────────────────────────────────────────────

def apply_dustin_tone(formatted_answer: str, client=None) -> str:
    """
    Applies Dustin Henderson's personality to the formatted answer.

    WHAT THIS LAYER DOES:
    Sends the clean answer to Claude with a tight persona prompt that
    rewrites the tone while preserving every factual detail.

    DUSTIN TONE RULES (enforced in the prompt):
    1. Do NOT change factual meaning — same information, different voice
    2. Do NOT add new information — zero hallucination risk
    3. Keep explanation clear and structured — no sacrifice of clarity
    4. Add light conversational phrasing — 1-2 sentences max per section
    5. Use analogies sparingly — only when they genuinely help
    6. No excessive slang, emojis, or over-casual language
    7. 80% professional, 20% personality

    WHY A SEPARATE API CALL FOR TONE?
    Doing tone transformation in the same call as retrieval + grounding
    creates conflicting objectives — the grounding rules say "be precise"
    while the tone rules say "be conversational". Separating them lets each
    Claude call do one job perfectly. The cost is one extra API call, but
    the quality difference is significant.

    SAFETY: The tone prompt explicitly forbids adding content. The only
    change allowed is phrasing. This makes it impossible for the tone layer
    to introduce hallucinations.
    """

    tone_system = """You are applying a light personality layer to a technical answer.

The character is Dustin Henderson from Stranger Things — enthusiastic, smart, 
uses clear analogies when helpful, genuinely excited about technology, but 
fundamentally a precise and reliable explainer.

STRICT RULES — violating these is not allowed:
1. DO NOT change any factual content — same facts, same structure
2. DO NOT add any new information — zero additions
3. DO NOT remove any technical detail — keep everything
4. ADD only: a warm opening phrase, occasional "here's the thing" transitions,
   one light analogy per section if it genuinely helps understanding
5. NO excessive slang, NO emojis, NO "dude/bro/man" more than once total
6. 80% of words must be the original professional content
7. Keep all section headers and the answer structure intact
8. Sources section: leave completely unchanged — do not touch it

Transform the tone, not the content."""

    # Gemini API: genai.GenerativeModel wraps the model.
    # generate_content() takes a single string — we combine system + user
    # into one prompt since Gemini's basic API doesn't have separate roles.
    # response.text → the generated string.
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"Apply Dustin tone to this answer:\n\n{formatted_answer}",
        config=types.GenerateContentConfig(
            system_instruction=tone_system,
        )
    )
    return response.text.strip()


# ── LAYER 5: OUTPUT MODES ─────────────────────────────────────────────────────

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
    print("\n" + "═" * 64)
    print(f"  {'🔧 DEBUG MODE' if mode == 'debug' else '💬 NVIDIA AI Advisor'}")
    print(f"  Q: {output.query}")
    print("═" * 64)

    # ── DEBUG: show retrieval diagnostics ────────────────────────────────────
    if mode == "debug":
        r = output.retrieval
        print(f"\n  RETRIEVAL DIAGNOSTICS")
        print(f"  ─────────────────────")
        print(f"  Detected tool  : {r.detected_tool or 'None (general query)'}")
        print(f"  Top score      : {r.top_score:.4f}  "
              f"{'✅ passed' if r.passed_check else '❌ failed'} "
              f"(threshold: {QUALITY_THRESHOLD})")

        print(f"\n  Retrieved chunks:")
        for i, chunk in enumerate(r.chunks, 1):
            score   = chunk.get("similarity_score", 0)
            bar_len = min(int(score * 20), 20)
            bar     = "█" * bar_len + "░" * (20 - bar_len)
            print(f"\n  #{i}  [{chunk['chunk_id']}]  score={score:.4f}  {bar}")
            print(f"       Source  : {chunk['source'].upper()}")
            print(f"       Section : {chunk['section']}")
            print(f"       Preview : {chunk['content'][:120].replace(chr(10), ' ')}...")

        print(f"\n  ─────────────────────")
        print(f"  RAW ANSWER (from Claude, before formatting):\n")
        for line in output.raw_answer.split("\n"):
            print(f"    {line}")

        print(f"\n  ─────────────────────")
        print(f"  FORMATTED ANSWER (after format_answer(), before tone):\n")
        for line in output.formatted_answer.split("\n"):
            print(f"    {line}")

        print(f"\n  ─────────────────────")
        print(f"  FINAL ANSWER (after apply_dustin_tone()):\n")

    # ── BOTH MODES: show final answer ─────────────────────────────────────────
    print()
    for line in output.final_answer.split("\n"):
        print(f"  {line}")

    # ── BOTH MODES: show sources ──────────────────────────────────────────────
    print(f"\n  {'─' * 60}")
    print(f"  SOURCES")
    for src in output.sources:
        score_indicator = f"(score: {src['score']:.2f})" if mode == "debug" else ""
        print(f"  • [{src['source']}] {src['section']} {score_indicator}")
        print(f"    {src['doc_link']}")

    print("\n" + "═" * 64)


# ─────────────────────────────────────────────────────────────────────────────
# NOT-FOUND RESPONSE — returned when quality check fails
# ─────────────────────────────────────────────────────────────────────────────

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
        "• NeMo    : https://docs.nvidia.com/nemo-framework/user-guide/latest/index.html\n"
        "• Triton  : https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html\n"
        "• TensorRT: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html"
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


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER PIPELINE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    query: str,
    index: faiss.Index,
    chunks: list[dict],
    embed_model: SentenceTransformer,
    client: None,  # not used — Gemini uses genai.GenerativeModel() per call
    mode: str = OUTPUT_MODE,
) -> PipelineOutput:
    """
    Runs the full RAG pipeline for a single query.

    STAGE ORDER (with reasons):

    1. retrieve_chunks()      — get candidate chunks from FAISS
    2. quality_check()        — gate: is this query answerable from our docs?
    3. build_grounded_prompt()— inject chunks + strict grounding rules
    4. generate (Claude API)  — get raw answer grounded in context
    5. format_answer()        — split answer body from sources, clean up
    6. apply_dustin_tone()    — transform tone without touching facts
    7. return PipelineOutput  — clean container for the display layer

    If stage 2 fails, we skip stages 3-6 entirely and return not_found_output().
    """

    # ── Pre-stage: conversational detection ─────────────────────────────────
    # Greetings, goodbyes, thanks → skip retrieval entirely.
    # LLM responds naturally as Dustin. No quality gate. No chunks needed.
    if is_conversational(query):
        return handle_conversational(query, mode, client)

    # ── Pre-stage: NVIDIA-adjacent detection ────────────────────────────────
    # Query is about NVIDIA broadly but outside NeMo/Triton/TensorRT scope.
    # Dustin redirects naturally to official NVIDIA docs with the link.
    if is_nvidia_adjacent(query):
        return handle_nvidia_adjacent(query, mode, client)

    # Stage 1 — Retrieve
    retrieval = retrieve_chunks(query, index, chunks, embed_model)

    # Stage 2 — Quality gate
    retrieval = quality_check(retrieval)
    if not retrieval.passed_check:
        return not_found_output(query, retrieval, mode)

    # Stage 3 — Build grounded prompt
    system_prompt, user_message = build_grounded_prompt(query, retrieval)

    # Stage 4 — Generate raw answer
    # Gemini: system_instruction sets grounding rules, generate_content()
    # takes the user message. response.text → the answer string.
    raw_answer = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
        )
    ).text

    # Stage 5 — Format answer
    formatted_answer, sources = format_answer(raw_answer, retrieval)

    # Stage 6 — Apply Dustin tone
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


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print("  NVIDIA AI System Advisor v2")
    print("  MiniLM + FAISS + Gemini 1.5 Flash + Dustin")
    print(f"  Mode: {OUTPUT_MODE.upper()}")
    print("=" * 64)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "\nGEMINI_API_KEY not set.\n"
            "Add to .env file: GEMINI_API_KEY=AIza...\n"
            "Get your free key at: https://aistudio.google.com\n"
        )

    # Configure Gemini globally — all genai.GenerativeModel() calls
    # automatically use this key after this line.
    client = genai.Client(api_key=api_key)

    # Load all artifacts at startup
    index, chunks, embed_model, _ = load_artifacts()
    print(f"\n  ✅ Gemini ready  ({GEMINI_MODEL})")

    # Demo queries
    demo_queries = [
        "What is NeMo Platform and when should I use it?",
        "How does Triton handle dynamic batching?",
        "What is the difference between NeMo, Triton, and TensorRT?",
        "How do I make a pizza?",   # ← intentional out-of-domain query to test quality gate
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
            print("\n  Later! — Dustin")
            break

        output = run_pipeline(query, index, chunks, embed_model, client, mode=OUTPUT_MODE)
        print_output(output)


if __name__ == "__main__":
    main()