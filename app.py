"""
NVIDIA AI System Advisor — Streamlit Chat UI (v2)
==================================================
Enhanced version with:
  1. Styled response cards with type labels
  2. Redesigned sources section — button-style links
  3. Staged typing indicator with contextual messages
  4. Response type detection labels (explanation/recommendation/etc.)
  5. Follow-up suggestion buttons after each response
  6. Sticky header after welcome screen
  7. Debug mode badge in header
  8. Improved spacing, alignment, visual hierarchy

Run with:
    streamlit run app.py

Backend unchanged — imports only load_artifacts, run_pipeline from rag_pipeline.py
"""

import streamlit as st
import os
import time
import random
from dotenv import load_dotenv

load_dotenv()

from rag_pipeline import load_artifacts, run_pipeline
from google import genai

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Dustin | NVIDIA AI Advisor",
    page_icon="🎮",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════════
# DUSTIN PIXEL ART — 16×16 SVG
# ═══════════════════════════════════════════════════════════════════════════════

DUSTIN_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="48" height="48" style="image-rendering:pixelated;display:block;">
  <rect x="3" y="1" width="10" height="2" fill="#c0392b"/>
  <rect x="2" y="2" width="12" height="1" fill="#c0392b"/>
  <rect x="1" y="3" width="14" height="1" fill="#e74c3c"/>
  <rect x="1" y="4" width="14" height="1" fill="#2c3e50"/>
  <rect x="2" y="5" width="12" height="6" fill="#f0c080"/>
  <rect x="4" y="6" width="2" height="2" fill="#2c3e50"/>
  <rect x="10" y="6" width="2" height="2" fill="#2c3e50"/>
  <rect x="5" y="6" width="1" height="1" fill="#fff"/>
  <rect x="11" y="6" width="1" height="1" fill="#fff"/>
  <rect x="7" y="8" width="2" height="1" fill="#d4956a"/>
  <rect x="4" y="9" width="8" height="1" fill="#2c3e50"/>
  <rect x="3" y="10" width="10" height="1" fill="#2c3e50"/>
  <rect x="4" y="9" width="2" height="1" fill="#fff"/>
  <rect x="7" y="9" width="2" height="1" fill="#fff"/>
  <rect x="10" y="9" width="1" height="1" fill="#fff"/>
  <rect x="3" y="11" width="10" height="1" fill="#f0c080"/>
  <rect x="6" y="12" width="4" height="1" fill="#f0c080"/>
  <rect x="2" y="13" width="12" height="3" fill="#2980b9"/>
  <rect x="5" y="12" width="6" height="1" fill="#fff"/>
</svg>"""

# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE TYPE DETECTION
# Classifies each response so we can show a label on the card
# ═══════════════════════════════════════════════════════════════════════════════

def detect_response_type(query: str, answer: str) -> tuple[str, str]:
    """
    Returns (label, color) for the response type badge.
    Detected from query intent — no extra API call needed.

    Types:
      EXPLANATION     — "what is", "how does", "explain"
      RECOMMENDATION  — "which", "should I", "when to use"
      COMPARISON      — "difference", "vs", "compare"
      GREETING        — casual conversation
      REDIRECT        — nvidia-adjacent but out of scope
      GENERAL         — fallback
    """
    q = query.lower()
    if any(w in q for w in ["what is", "what are", "explain", "how does", "how do", "what does"]):
        return "EXPLANATION", "#2980b9"
    if any(w in q for w in ["which", "should i", "when to use", "recommend", "better", "best"]):
        return "RECOMMENDATION", "#27ae60"
    if any(w in q for w in ["difference", " vs ", "compare", "versus", "between"]):
        return "COMPARISON", "#8e44ad"
    if any(w in q for w in ["hey", "hi", "hello", "bye", "thanks", "do you copy", "you there"]):
        return "GREETING", "#e67e22"
    if "nvidia" in q and not any(w in q for w in ["nemo", "triton", "tensorrt"]):
        return "REDIRECT", "#7f8c8d"
    return "GENERAL", "#c0392b"


# ═══════════════════════════════════════════════════════════════════════════════
# FOLLOW-UP SUGGESTION GENERATOR
# Returns 3 contextual follow-up questions based on what was just answered
# ═══════════════════════════════════════════════════════════════════════════════

FOLLOWUPS = {
    "nemo": [
        "How does NeMo handle model evaluation?",
        "What is the NeMo Data Designer service?",
        "How do NeMo Guardrails work?",
        "What storage backends does NeMo support?",
        "How does NeMo fine-tuning work with LoRA?",
    ],
    "triton": [
        "How does Triton's dynamic batching work?",
        "What backends does Triton support?",
        "How do I deploy Triton on Kubernetes?",
        "What is Business Logic Scripting in Triton?",
        "How does Triton handle concurrent model execution?",
    ],
    "tensorrt": [
        "What is the difference between TensorRT build and runtime phase?",
        "How does TRT-LLM handle KV cache?",
        "What quantization options does TensorRT support?",
        "How do I convert a PyTorch model to TensorRT?",
        "What is speculative decoding in TRT-LLM?",
    ],
    "general": [
        "What is the difference between NeMo, Triton, and TensorRT?",
        "When should I use NeMo vs Triton?",
        "How do TensorRT and Triton work together?",
        "What is the full NVIDIA AI production stack?",
        "How does TRT-LLM integrate with Triton?",
    ],
}

def get_followups(output) -> list[str]:
    """
    Picks 3 relevant follow-up questions based on which tool was retrieved.
    Falls back to general questions for conversational responses.
    """
    tool = output.retrieval.detected_tool
    pool = FOLLOWUPS.get(tool, FOLLOWUPS["general"])
    # Don't suggest the same question that was just asked
    filtered = [q for q in pool if q.lower() not in st.session_state.get("asked", set())]
    return random.sample(filtered, min(3, len(filtered)))


# ═══════════════════════════════════════════════════════════════════════════════
# STAGED TYPING MESSAGES
# Shows contextual status messages while the pipeline runs
# ═══════════════════════════════════════════════════════════════════════════════

TYPING_STAGES = {
    "nemo":     ["Scanning NeMo docs...", "Retrieving platform specs...", "Asking Dustin..."],
    "triton":   ["Checking Triton architecture...", "Pulling serving docs...", "Asking Dustin..."],
    "tensorrt": ["Digging into TensorRT internals...", "Fetching optimization specs...", "Asking Dustin..."],
    "general":  ["Searching the knowledge base...", "Matching your query...", "Asking Dustin..."],
    "chat":     ["Dustin is typing...", "Warming up the radios...", "Almost..."],
}

def get_typing_stages(query: str) -> list[str]:
    q = query.lower()
    if any(w in q for w in ["hey","hi","hello","bye","thanks","do you copy"]):
        return TYPING_STAGES["chat"]
    if "nemo" in q: return TYPING_STAGES["nemo"]
    if "triton" in q: return TYPING_STAGES["triton"]
    if any(w in q for w in ["tensorrt","trt"]): return TYPING_STAGES["tensorrt"]
    return TYPING_STAGES["general"]


# ═══════════════════════════════════════════════════════════════════════════════
# CSS — retro pixel theme, enhanced components
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Share+Tech+Mono&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    background-color: #0a0a0f !important;
    color: #e8e8e8 !important;
}
.stApp {
    background: #0a0a0f !important;
    background-image:
        radial-gradient(ellipse at 15% 50%, rgba(192,57,43,0.06) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 15%, rgba(41,128,185,0.04) 0%, transparent 55%);
}
#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none; }

/* ── CRT scanlines ── */
.stApp::before {
    content:"";
    position:fixed; top:0; left:0; width:100%; height:100%;
    background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.025) 2px,rgba(0,0,0,0.025) 4px);
    pointer-events:none; z-index:9998;
}

/* ══════════════════════════════════════
   STICKY HEADER (shown after welcome)
══════════════════════════════════════ */
.sticky-header {
    position: sticky;
    top: 0;
    z-index: 100;
    background: rgba(10,10,15,0.95);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid rgba(192,57,43,0.25);
    padding: 0.6rem 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
}
.sticky-header-title {
    font-family: 'Press Start 2P', monospace;
    font-size: 0.55rem;
    color: #e74c3c;
    text-shadow: 0 0 8px rgba(231,76,60,0.6);
    letter-spacing: 0.08em;
    flex: 1;
}
.sticky-header-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem;
    color: #555;
    letter-spacing: 0.05em;
}
.debug-badge {
    font-family: 'Press Start 2P', monospace;
    font-size: 0.4rem;
    background: rgba(192,57,43,0.2);
    border: 1px solid #c0392b;
    color: #e74c3c;
    padding: 3px 7px;
    border-radius: 2px;
    animation: flicker 2s infinite;
}

/* ══════════════════════════════════════
   WELCOME CARD
══════════════════════════════════════ */
.welcome-card {
    background: linear-gradient(135deg, #0d1117 0%, #1a0a0a 100%);
    border: 1px solid #c0392b;
    border-radius: 4px;
    padding: 2.5rem 2rem 2rem;
    text-align: center;
    margin: 0.5rem 0 1.5rem;
    box-shadow: 0 0 40px rgba(192,57,43,0.15), inset 0 0 40px rgba(0,0,0,0.4);
    position: relative;
    overflow: hidden;
}
.welcome-card::after {
    content:"";
    position:absolute; top:-50%; left:-50%;
    width:200%; height:200%;
    background:radial-gradient(ellipse, rgba(192,57,43,0.04) 0%, transparent 65%);
    animation:pulse-bg 5s ease-in-out infinite;
}
@keyframes pulse-bg { 0%,100%{opacity:.4} 50%{opacity:1} }

.pixel-wrap {
    display:inline-block;
    border:2px solid #c0392b;
    box-shadow:0 0 20px rgba(192,57,43,0.5), 0 0 40px rgba(192,57,43,0.2);
    padding:6px;
    background:#110505;
    image-rendering:pixelated;
    margin-bottom:1rem;
}
.welcome-title {
    font-family:'Press Start 2P',monospace;
    font-size:0.9rem;
    color:#e74c3c;
    text-shadow:0 0 12px rgba(231,76,60,0.9),0 0 24px rgba(231,76,60,0.4);
    margin:0.5rem 0 0.4rem;
    animation:flicker 4s infinite;
}
.welcome-sub {
    font-family:'Share Tech Mono',monospace;
    font-size:0.75rem;
    color:#7fb3d3;
    letter-spacing:0.1em;
    margin-bottom:1rem;
}
.welcome-desc {
    font-family:'Share Tech Mono',monospace;
    font-size:0.71rem;
    color:#999;
    line-height:1.9;
    max-width:480px;
    margin:0 auto;
}
@keyframes flicker {
    0%,94%,100%{opacity:1} 95%{opacity:.7} 97%{opacity:.5} 99%{opacity:.9}
}

/* ══════════════════════════════════════
   SECTION LABELS
══════════════════════════════════════ */
.section-label {
    font-family:'Press Start 2P',monospace;
    font-size:0.48rem;
    color:#444;
    letter-spacing:0.12em;
    text-transform:uppercase;
    margin:1rem 0 0.5rem;
}

/* ══════════════════════════════════════
   QUICK START BUTTONS
══════════════════════════════════════ */
.stButton button {
    background:transparent !important;
    border:1px solid rgba(192,57,43,0.4) !important;
    color:#c0392b !important;
    font-family:'Share Tech Mono',monospace !important;
    font-size:0.65rem !important;
    border-radius:2px !important;
    padding:0.3rem 0.5rem !important;
    transition:all 0.15s !important;
    line-height:1.4 !important;
}
.stButton button:hover {
    background:rgba(192,57,43,0.12) !important;
    border-color:#e74c3c !important;
    color:#e74c3c !important;
    box-shadow:0 0 8px rgba(231,76,60,0.25) !important;
}

/* ══════════════════════════════════════
   CHAT MESSAGES — base
══════════════════════════════════════ */
.stChatMessage { background:transparent !important; border:none !important; padding:0.25rem 0 !important; }
.stChatMessage p, .stChatMessage li, .stChatMessage span {
    font-family:'Share Tech Mono',monospace !important;
    font-size:0.8rem !important;
    line-height:1.85 !important;
    color:#ddd !important;
}

/* ══════════════════════════════════════
   RESPONSE CARD
══════════════════════════════════════ */
.response-card {
    background: linear-gradient(160deg, rgba(15,15,25,0.95) 0%, rgba(20,8,8,0.9) 100%);
    border: 1px solid rgba(192,57,43,0.22);
    border-left: 3px solid #c0392b;
    border-radius: 0 4px 4px 0;
    padding: 1rem 1.25rem 0.875rem;
    margin: 0.25rem 0 0.5rem;
    position: relative;
}
.response-card::before {
    content:"";
    position:absolute; top:0; right:0;
    width:60px; height:100%;
    background:linear-gradient(90deg,transparent,rgba(192,57,43,0.03));
    border-radius:0 4px 4px 0;
}

/* ── Response type badge ── */
.type-badge {
    display:inline-block;
    font-family:'Press Start 2P',monospace;
    font-size:0.4rem;
    padding:3px 8px;
    border-radius:2px;
    letter-spacing:0.08em;
    margin-bottom:0.75rem;
    border:1px solid currentColor;
}

/* ── Response text inside card ── */
.response-card p, .response-card li {
    font-family:'Share Tech Mono',monospace !important;
    font-size:0.8rem !important;
    line-height:1.9 !important;
    color:#d8d8d8 !important;
    margin:0.3rem 0 !important;
}

/* ══════════════════════════════════════
   SOURCES SECTION — redesigned
══════════════════════════════════════ */
.sources-wrap {
    margin-top:0.875rem;
    padding-top:0.75rem;
    border-top:1px solid rgba(192,57,43,0.15);
}
.sources-label {
    font-family:'Press Start 2P',monospace;
    font-size:0.42rem;
    color:#555;
    letter-spacing:0.1em;
    margin-bottom:0.5rem;
}
.source-item {
    display:flex;
    align-items:flex-start;
    gap:0.6rem;
    margin:0.4rem 0;
    padding:0.45rem 0.7rem;
    background:rgba(0,0,0,0.3);
    border:1px solid rgba(255,255,255,0.05);
    border-radius:2px;
    transition:border-color 0.15s;
}
.source-item:hover { border-color:rgba(192,57,43,0.3); }
.source-tag {
    font-family:'Press Start 2P',monospace;
    font-size:0.38rem;
    padding:3px 6px;
    border-radius:2px;
    white-space:nowrap;
    margin-top:2px;
    flex-shrink:0;
}
.source-tag-nemo    { background:rgba(39,174,96,0.15);  color:#2ecc71; border:1px solid rgba(39,174,96,0.3); }
.source-tag-triton  { background:rgba(41,128,185,0.15); color:#3498db; border:1px solid rgba(41,128,185,0.3); }
.source-tag-tensorrt{ background:rgba(155,89,182,0.15); color:#9b59b6; border:1px solid rgba(155,89,182,0.3); }
.source-tag-default { background:rgba(192,57,43,0.15);  color:#e74c3c; border:1px solid rgba(192,57,43,0.3); }
.source-text { flex:1; }
.source-section {
    font-family:'Share Tech Mono',monospace;
    font-size:0.68rem;
    color:#bbb;
    display:block;
    margin-bottom:2px;
}
.source-link {
    font-family:'Share Tech Mono',monospace;
    font-size:0.6rem;
    color:#555;
    text-decoration:none;
    border-bottom:1px solid transparent;
    transition:all 0.15s;
}
.source-link:hover { color:#888; border-bottom-color:#666; }

/* ══════════════════════════════════════
   FOLLOW-UP SUGGESTIONS
══════════════════════════════════════ */
.followup-wrap {
    margin-top:0.75rem;
    padding-top:0.6rem;
    border-top:1px solid rgba(255,255,255,0.05);
}
.followup-label {
    font-family:'Press Start 2P',monospace;
    font-size:0.4rem;
    color:#444;
    letter-spacing:0.1em;
    margin-bottom:0.4rem;
}
/* Follow-up buttons get a slightly different style */
.followup-wrap .stButton button {
    font-size:0.61rem !important;
    border-color:rgba(255,255,255,0.08) !important;
    color:#777 !important;
    text-align:left !important;
    padding:0.35rem 0.6rem !important;
}
.followup-wrap .stButton button:hover {
    border-color:rgba(192,57,43,0.4) !important;
    color:#c0392b !important;
    background:rgba(192,57,43,0.06) !important;
    box-shadow:none !important;
}

/* ══════════════════════════════════════
   TYPING INDICATOR
══════════════════════════════════════ */
.typing-wrap {
    display:flex;
    align-items:center;
    gap:0.5rem;
    padding:0.5rem 0;
}
.typing-msg {
    font-family:'Share Tech Mono',monospace;
    font-size:0.68rem;
    color:#555;
    animation:fadein 0.3s ease;
}
@keyframes fadein { from{opacity:0;transform:translateY(3px)} to{opacity:1;transform:translateY(0)} }
.typing-dots { display:flex; gap:4px; }
.dot {
    width:5px; height:5px;
    background:#c0392b;
    border-radius:50%;
    animation:bounce 1.2s ease-in-out infinite;
    box-shadow:0 0 4px rgba(192,57,43,0.5);
}
.dot:nth-child(2){animation-delay:.2s}
.dot:nth-child(3){animation-delay:.4s}
@keyframes bounce {
    0%,80%,100%{transform:translateY(0);opacity:.4}
    40%{transform:translateY(-5px);opacity:1}
}

/* ══════════════════════════════════════
   CHAT INPUT
══════════════════════════════════════ */
.stChatInputContainer {
    border-top:1px solid rgba(192,57,43,0.2) !important;
    background:#0a0a0f !important;
    padding-top:0.5rem !important;
}
.stChatInput textarea {
    background:#0c0c18 !important;
    border:1px solid rgba(192,57,43,0.35) !important;
    border-radius:2px !important;
    color:#e8e8e8 !important;
    font-family:'Share Tech Mono',monospace !important;
    font-size:0.78rem !important;
    caret-color:#e74c3c !important;
}
.stChatInput textarea:focus {
    border-color:#e74c3c !important;
    box-shadow:0 0 10px rgba(231,76,60,0.2) !important;
    outline:none !important;
}

/* ══════════════════════════════════════
   SIDEBAR
══════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background:#0c0c14 !important;
    border-right:1px solid rgba(192,57,43,0.15) !important;
}
.stToggle label { font-family:'Share Tech Mono',monospace !important; font-size:0.7rem !important; }

/* ══════════════════════════════════════
   DISCLAIMER
══════════════════════════════════════ */
.disclaimer {
    font-family:'Share Tech Mono',monospace;
    font-size:0.59rem;
    color:#444;
    text-align:center;
    padding:0.75rem 1.5rem 0.5rem;
    border-top:1px solid rgba(255,255,255,0.04);
    line-height:1.8;
    margin-top:0.5rem;
}
.disclaimer a { color:#666; }
.disclaimer strong { color:#666; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

defaults = {
    "messages":    [],
    "show_welcome": True,
    "debug_mode":  False,
    "asked":       set(),
    "last_followups": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════════
# CACHED ARTIFACT LOADER
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_artifacts():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None, None, None, None, None
    client = genai.Client(api_key=api_key)
    index, chunks, embed_model, metadata = load_artifacts()
    return index, chunks, embed_model, metadata, client


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS — render response card, sources, follow-ups
# ═══════════════════════════════════════════════════════════════════════════════

def source_tag_class(source: str) -> str:
    mapping = {"NEMO": "nemo", "TRITON": "triton", "TENSORRT": "tensorrt"}
    return f"source-tag source-tag-{mapping.get(source.upper(), 'default')}"


def render_response_card(answer: str, query: str):
    """Renders the styled response card with type badge."""
    label, color = detect_response_type(query, answer)
    st.markdown(
        f"""<div class="response-card">
            <span class="type-badge" style="color:{color};border-color:{color};">
                {label}
            </span>
            <div class="response-text">{answer}</div>
        </div>""",
        unsafe_allow_html=True
    )


def render_sources(sources: list[dict]):
    """Renders the redesigned sources section with button-style items."""
    if not sources:
        return
    html = '<div class="sources-wrap"><div class="sources-label">// sources accessed</div>'
    for src in sources:
        tag_cls = source_tag_class(src["source"])
        html += f"""
        <div class="source-item">
            <span class="{tag_cls}">{src['source']}</span>
            <div class="source-text">
                <span class="source-section">{src['section']}</span>
                <a class="source-link" href="{src['doc_link']}" target="_blank">
                    ↗ {src['doc_link'][:55]}{'...' if len(src['doc_link']) > 55 else ''}
                </a>
            </div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_followups(followups: list[str]):
    """Renders follow-up suggestion buttons below a response."""
    if not followups:
        return
    st.markdown('<div class="followup-wrap"><div class="followup-label">// ask next</div>', unsafe_allow_html=True)
    cols = st.columns(len(followups))
    for i, (col, q) in enumerate(zip(cols, followups)):
        with col:
            if st.button(f"↳ {q}", key=f"fu_{hash(q)}_{len(st.session_state.messages)}", use_container_width=True):
                st.session_state.pending_prompt = q
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def render_debug(output) -> str:
    """Builds debug text and renders the expander. Returns debug_text for history."""
    debug_text = ""
    if st.session_state.debug_mode and output.retrieval.chunks:
        r = output.retrieval
        debug_text  = f"Detected tool : {r.detected_tool or 'None (general)'}\n"
        debug_text += f"Top score     : {r.top_score:.4f} "
        debug_text += f"({'PASS' if r.passed_check else 'FAIL'}, threshold=0.25)\n\n"
        for i, chunk in enumerate(r.chunks, 1):
            score = chunk.get("similarity_score", 0)
            debug_text += f"#{i} [{chunk['chunk_id']}]  score={score:.4f}\n"
            debug_text += f"   {chunk['section']}\n"
            debug_text += f"   {chunk['content'][:80].replace(chr(10),' ')}...\n\n"
        with st.expander("🔧 debug", expanded=False):
            st.code(debug_text, language="text")
    return debug_text


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="font-family:'Press Start 2P',monospace;font-size:0.5rem;
         color:#e74c3c;text-shadow:0 0 8px rgba(231,76,60,0.5);
         padding:1rem 0 0.75rem;text-align:center;letter-spacing:0.08em;">
    ⚙ SYSTEM STATUS
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.66rem;
         color:#999;line-height:2.4;padding:0.25rem 0;">
    <span style="color:#2ecc71">▪</span> Embeddings · MiniLM-L6-v2<br>
    <span style="color:#2ecc71">▪</span> Vector DB  · FAISS · 57 chunks<br>
    <span style="color:#2ecc71">▪</span> LLM        · Gemini 2.5 Flash Lite<br>
    <span style="color:#2ecc71">▪</span> Persona    · Dustin Henderson<br>
    <span style="color:#f39c12">▪</span> Scope      · NeMo · Triton · TRT
    </div>""", unsafe_allow_html=True)

    st.divider()

    st.markdown("""<div style="font-family:'Press Start 2P',monospace;font-size:0.45rem;
         color:#444;margin-bottom:0.5rem;">DEV OPTIONS</div>""", unsafe_allow_html=True)

    debug = st.toggle("🔧 Debug mode", value=st.session_state.debug_mode)
    st.session_state.debug_mode = debug

    if debug:
        st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;
             color:#666;line-height:1.9;margin-top:0.35rem;">
        Shows scores · chunk IDs<br>source routing · raw retrieval
        </div>""", unsafe_allow_html=True)

    st.divider()

    if st.button("🗑  Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.show_welcome = True
        st.session_state.asked = set()
        st.rerun()

    st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:0.56rem;
         color:#333;text-align:center;margin-top:1.5rem;line-height:2;">
    NVIDIA AI Advisor v2<br>RAG · MiniLM · FAISS · Gemini
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STICKY HEADER — shown once conversation has started
# ═══════════════════════════════════════════════════════════════════════════════

if not st.session_state.show_welcome or len(st.session_state.messages) > 0:
    debug_badge = '<span class="debug-badge">DEBUG ON</span>' if st.session_state.debug_mode else ""
    st.markdown(
        f"""<div class="sticky-header">
            <div style="image-rendering:pixelated;flex-shrink:0;">{DUSTIN_SVG}</div>
            <div style="flex:1;">
                <div class="sticky-header-title">DUSTIN · NVIDIA AI ADVISOR {debug_badge}</div>
                <div class="sticky-header-sub">NeMo · Triton · TensorRT</div>
            </div>
        </div>""",
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
# WELCOME SCREEN
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.show_welcome and len(st.session_state.messages) == 0:
    st.markdown(f"""
    <div class="welcome-card">
        <div class="pixel-wrap">{DUSTIN_SVG}</div>
        <div class="welcome-title">MEET DUSTIN</div>
        <div class="welcome-sub">YOUR NVIDIA AI ECOSYSTEM GUIDE</div>
        <div class="welcome-desc">
            Hey! Dustin Henderson here — and yes, I've got the whole
            NVIDIA AI stack loaded up. NeMo Platform, Triton Inference
            Server, TensorRT and TRT-LLM — ask me anything about these
            tools and I'll break it down for you, no Demogorgons included.
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Greeting row ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">// greet dustin</div>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    greetings = [
        ("👋 Hey Dustin, you there?",  "Hey Dustin, you there?"),
        ("📡 Do you copy, Dustin?",     "Do you copy, Dustin?"),
        ("🎮 Hello there!",             "Hello there, Dustin!"),
    ]
    for col, (label, prompt) in zip([g1, g2, g3], greetings):
        with col:
            if st.button(label, use_container_width=True, key=f"greet_{label}"):
                st.session_state.pending_prompt = prompt
                st.rerun()

    # ── Topic row ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:1.1rem;">// jump straight in</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    topics = [
        ("🧠 What is NeMo?",            "What is NeMo Platform and when should I use it?"),
        ("⚡ How does Triton work?",     "How does Triton Inference Server work?"),
        ("🔧 What is TensorRT?",        "What is TensorRT and how does it optimize models?"),
    ]
    for col, (label, prompt) in zip([t1, t2, t3], topics):
        with col:
            if st.button(label, use_container_width=True, key=f"topic_{label}"):
                st.session_state.pending_prompt = prompt
                st.rerun()

    # ── Goodbye row ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:1.1rem;">// wrap up a session</div>', unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    byes = [
        ("🙏 Thanks, bye Dustin!", "Thanks Dustin, that was super helpful. Bye!"),
        ("👋 See you later!",       "See you later, Dustin!"),
        ("✌  Later!",              "Later, Dustin!"),
    ]
    for col, (label, prompt) in zip([b1, b2, b3], byes):
        with col:
            if st.button(label, use_container_width=True, key=f"bye_{label}"):
                st.session_state.pending_prompt = prompt
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# CHAT HISTORY — render previous messages
# ═══════════════════════════════════════════════════════════════════════════════

for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar=DUSTIN_SVG):
            # Response card with type badge
            label, color = detect_response_type(
                st.session_state.messages[i-1]["content"] if i > 0 else "",
                msg["content"]
            )
            st.markdown(
                f"""<div class="response-card">
                    <span class="type-badge" style="color:{color};border-color:{color};">{label}</span>
                    <div>{msg['content']}</div>
                </div>""",
                unsafe_allow_html=True
            )
            # Sources
            if msg.get("sources"):
                render_sources(msg["sources"])
            # Debug
            if msg.get("debug_info") and st.session_state.debug_mode:
                with st.expander("🔧 debug", expanded=False):
                    st.code(msg["debug_info"], language="text")
            # Follow-ups (only show for last message)
            if i == len(st.session_state.messages) - 1 and msg.get("followups"):
                render_followups(msg["followups"])


# ═══════════════════════════════════════════════════════════════════════════════
# HANDLE PENDING PROMPT (button clicks)
# ═══════════════════════════════════════════════════════════════════════════════

prompt = st.session_state.pop("pending_prompt", None)
user_input = st.chat_input("Ask Dustin anything about NeMo, Triton, or TensorRT...")
if user_input:
    prompt = user_input


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESS QUERY
# ═══════════════════════════════════════════════════════════════════════════════

if prompt:
    st.session_state.show_welcome = False
    st.session_state.asked.add(prompt.lower())

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    index, chunks, embed_model, metadata, client = get_artifacts()
    if client is None:
        st.error("⚠ GEMINI_API_KEY not found in .env file.")
        st.stop()

    with st.chat_message("assistant", avatar=DUSTIN_SVG):
        typing_ph = st.empty()

        # ── Staged typing indicator ───────────────────────────────────────────
        stages = get_typing_stages(prompt)
        for stage_msg in stages:
            typing_ph.markdown(
                f"""<div class="typing-wrap">
                    <div class="typing-dots">
                        <div class="dot"></div>
                        <div class="dot"></div>
                        <div class="dot"></div>
                    </div>
                    <span class="typing-msg">{stage_msg}</span>
                </div>""",
                unsafe_allow_html=True
            )
            time.sleep(0.6)

        # ── Run pipeline ──────────────────────────────────────────────────────
        mode = "debug" if st.session_state.debug_mode else "user"
        output = run_pipeline(prompt, index, chunks, embed_model, client, mode=mode)
        typing_ph.empty()

        # ── Response card with type badge ─────────────────────────────────────
        label, color = detect_response_type(prompt, output.final_answer)
        st.markdown(
            f"""<div class="response-card">
                <span class="type-badge" style="color:{color};border-color:{color};">{label}</span>
                <div>{output.final_answer}</div>
            </div>""",
            unsafe_allow_html=True
        )

        # ── Sources ───────────────────────────────────────────────────────────
        render_sources(output.sources)

        # ── Debug ─────────────────────────────────────────────────────────────
        debug_text = render_debug(output)

        # ── Follow-up suggestions ─────────────────────────────────────────────
        followups = get_followups(output)
        render_followups(followups)

    # Save to history
    st.session_state.messages.append({
        "role":       "assistant",
        "content":    output.final_answer,
        "sources":    output.sources,
        "debug_info": debug_text,
        "followups":  followups,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# DISCLAIMER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="disclaimer">
    ⚠ &nbsp;Dustin is explicitly trained on
    <strong>NVIDIA NeMo</strong>, <strong>Triton Inference Server</strong>,
    and <strong>TensorRT</strong> documentation only.
    Responses outside this scope may be incomplete or redirected to official docs.
    &nbsp;·&nbsp; This system can make mistakes — always verify critical information at
    <a href="https://docs.nvidia.com" target="_blank">docs.nvidia.com</a>.
</div>
""", unsafe_allow_html=True)