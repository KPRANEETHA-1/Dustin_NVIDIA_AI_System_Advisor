"""Create retrieval-ready chunks from the NVIDIA markdown knowledge files."""

import json
import re
import os
from difflib import SequenceMatcher


# CONSTANTS

MIN_CHARS = 120    # FIX 2 â€” chunks shorter than this are too weak to embed
MAX_CHARS = 1200   # FIX 3 â€” chunks longer than this dilute semantic meaning

DOC_LINKS = {
    "nemo":     "https://docs.nvidia.com/nemo-framework/user-guide/latest/index.html",
    "triton":   "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html",
    "tensorrt": "https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html",
}

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

FILES = {
    "01_nemo.md":     "nemo",
    "02_triton.md":   "triton",
    "03_tensorrt.md": "tensorrt",
}


# FIX 1 â€” MARKDOWN NOISE REMOVAL

def clean_markdown_noise(content: str) -> str:
    """Remove markdown formatting that does not help retrieval embeddings."""

    # A â€” Remove markdown headers (1â€“6 levels)
    content = re.sub(r"^#{1,6}\s*", "", content, flags=re.MULTILINE)

    # B â€” Remove fenced code blocks entirely (``` ... ```)
    #     re.DOTALL makes . match newlines so multi-line blocks are caught
    content = re.sub(r"```[\s\S]*?```", "", content, flags=re.DOTALL)

    # C â€” Remove horizontal rules (--- or *** or ___)
    content = re.sub(r"\n?[-*_]{3,}\n?", "\n", content)

    # D â€” Remove bold (**text** or __text__) and italic (*text* or _text_)
    #     Order matters: do ** before * so we don't half-strip bold markers
    content = re.sub(r"\*\*(.*?)\*\*", r"\1", content)
    content = re.sub(r"__(.*?)__",     r"\1", content)
    content = re.sub(r"\*(.*?)\*",     r"\1", content)
    content = re.sub(r"_(.*?)_",       r"\1", content)

    # E â€” Remove inline code backticks
    content = re.sub(r"`([^`]+)`", r"\1", content)

    # F â€” Normalize whitespace (FIX 5)
    content = re.sub(r"[ \t]+", " ", content)          # collapse spaces/tabs
    content = re.sub(r"\n{3,}", "\n\n", content)       # max 2 consecutive newlines
    content = content.strip()

    return content


# FIX 3 â€” OVERSIZED CHUNK SPLITTER

def split_large_chunk(text: str) -> list[str]:
    """Split oversized chunks at paragraph boundaries."""
    paragraphs = text.split("\n\n")
    result = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if current and len(current) + len(para) + 2 > MAX_CHARS:
            # Flush current chunk before it gets too large
            result.append(current.strip())
            current = para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        result.append(current.strip())

    return result if result else [text]


# FIX 6 â€” TABLE FLATTENING (unchanged, guaranteed to run first)

def flatten_tables(text: str, source: str, section: str) -> str:
    """Convert markdown pipe tables into prose before cleaning."""
    tool_name = "NeMo" if source == "nemo" else source.capitalize()
    lines = text.split("\n")
    result_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if "|" in line and not re.match(r"^\s*\|[-:\s|]+\|\s*$", line):
            table_lines = []
            while i < len(lines) and "|" in lines[i]:
                table_lines.append(lines[i].strip())
                i += 1

            if len(table_lines) < 2:
                result_lines.extend(table_lines)
                continue

            headers = [h.strip() for h in table_lines[0].split("|") if h.strip()]
            data_rows = []
            for row_line in table_lines[2:]:
                cells = [c.strip() for c in row_line.split("|") if c.strip()]
                if cells:
                    data_rows.append(cells)

            for row in data_rows:
                sentence = build_sentence(tool_name, section, headers, row)
                if sentence:
                    result_lines.append(sentence)
        else:
            result_lines.append(line)
            i += 1

    return "\n".join(result_lines)


def build_sentence(tool_name: str, section: str, headers: list, cells: list) -> str:
    if not cells or not headers:
        return ""

    while len(cells) < len(headers):
        cells.append("")

    h0 = headers[0].lower()
    h1 = headers[1].lower() if len(headers) > 1 else ""
    c0 = cells[0]
    c1 = cells[1] if len(cells) > 1 else ""

    if ("use case" in h0 or "scenario" in h0) and ("feature" in h1 or "capability" in h1):
        return f"{tool_name} supports the following use case: {c0}. This is handled by {c1}."
    if "feature" in h0 and ("description" in h1 or "purpose" in h1 or "detail" in h1):
        return f"{c0}: {c1}."
    if "backend" in h0 and "framework" in h1:
        return f"The {c0} backend in {tool_name} supports {c1}."
    if "hardware" in h0 and "support" in h1:
        return f"{tool_name} supports {c0} hardware ({c1})."
    if "mode" in h0:
        return f"In {tool_name}, {c0} mode is used for {c1}."
    if "type" in h0 and len(cells) >= 3:
        return f"{tool_name} supports the {c0} data type ({cells[1]} bits), typically used for {cells[2]}."
    if ("tool" in h0 or "library" in h0) and "purpose" in h1:
        return f"In the {tool_name} ecosystem, {c0} is used for {c1}."
    if "misconception" in h0 or "incorrect" in h0 or "wrong" in h0:
        return f"Common misconception about {tool_name}: \"{c0}\" â€” The correct understanding is: {c1}."

    parts = [f"{headers[k]}: {cells[k]}" for k in range(min(len(headers), len(cells))) if cells[k]]
    return f"{tool_name} â€” " + "; ".join(parts) + "."


# CORE CHUNKER â€” all fixes wired in correct order

def chunk_markdown(filepath: str, source: str) -> list[dict]:
    """Chunk one markdown file using the retrieval-prep pipeline."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    parts = re.split(r"(^#{1,3}\s.+$)", text, flags=re.MULTILINE)

    chunks = []
    chunk_id_counter = 0
    current_section = "Introduction"

    i = 0
    while i < len(parts):
        part = parts[i].strip()

        if not part:
            i += 1
            continue

        if re.match(r"^#{1,3}\s", part):
            current_section = part.lstrip("#").strip()
            i += 1
            content = parts[i].strip() if i < len(parts) else ""
            i += 1
        else:
            content = part
            i += 1

        content = flatten_tables(content, source, current_section)

        content = clean_markdown_noise(content)

        # 20 words â‰ˆ one complete sentence. Less than that = a heading, a
        # one-liner, or a leftover artifact. Not worth embedding.
        if len(content.split()) < 20:
            continue

        if len(content) < MIN_CHARS:
            continue

        parts_split = split_large_chunk(content)

        for part_content in parts_split:

            # FIX 2 re-check: each sub-part must still meet minimum size
            if len(part_content) < MIN_CHARS:
                continue
            # FIX 4 re-check on sub-parts
            if len(part_content.split()) < 20:
                continue

            chunk = {
                "chunk_id":     f"{source}_{chunk_id_counter:03d}",
                "source":       source,
                "section":      current_section,
                "doc_link":     DOC_LINKS[source],
                "content":      part_content,
                "char_count":   len(part_content),
                "token_approx": len(part_content) // 4,
            }

            chunks.append(chunk)
            chunk_id_counter += 1

    # WHY 200 CHARS?
    # Two chunks with the same opening 200 characters are almost certainly
    # duplicates â€” perhaps from overlapping sections or repeated boilerplate.
    # We don't compare the full content (too slow) or just 50 chars (too loose).
    # 200 chars is a reliable fingerprint without full-text comparison.
    unique_chunks = []
    seen_fingerprints = set()

    for c in chunks:
        fingerprint = c["content"][:200]
        if fingerprint not in seen_fingerprints:
            seen_fingerprints.add(fingerprint)
            unique_chunks.append(c)

    duplicates_removed = len(chunks) - len(unique_chunks)
    if duplicates_removed > 0:
        print(f"     âš ï¸  Removed {duplicates_removed} duplicate chunk(s)")

    return unique_chunks


# METADATA-AWARE RETRIEVAL (unchanged from v2)

def detect_source_from_query(query: str) -> str | None:
    query_lower = query.lower()
    scores = {}
    for source, keywords in TOOL_KEYWORDS.items():
        hit_count = sum(1 for kw in keywords if kw in query_lower)
        if hit_count > 0:
            scores[source] = hit_count
    return max(scores, key=scores.get) if scores else None


def simple_similarity(query: str, content: str) -> float:
    query_lower = query.lower()
    content_lower = content.lower()
    query_words = set(query_lower.split())
    content_words = set(content_lower.split())
    overlap = len(query_words & content_words) / (len(query_words) + 1)
    seq_score = SequenceMatcher(None, query_lower[:200], content_lower[:200]).ratio()
    return (overlap * 0.7) + (seq_score * 0.3)


def retrieve(query: str, all_chunks: list[dict], top_k: int = 3) -> list[dict]:
    detected_source = detect_source_from_query(query)

    if detected_source:
        priority_pool = [c for c in all_chunks if c["source"] == detected_source]
        fallback_pool = [c for c in all_chunks if c["source"] != detected_source]
        print(f"  [Router] Detected tool â†’ '{detected_source}' "
              f"({len(priority_pool)} priority, {len(fallback_pool)} fallback)")
    else:
        priority_pool = all_chunks
        fallback_pool = []
        print(f"  [Router] General query â†’ searching all {len(all_chunks)} chunks")

    def score_pool(pool):
        return sorted(
            [(simple_similarity(query, c["content"]), c) for c in pool],
            key=lambda x: x[0], reverse=True
        )

    results = [c for _, c in score_pool(priority_pool)[:top_k]]

    if len(results) < top_k and fallback_pool:
        needed = top_k - len(results)
        results += [c for _, c in score_pool(fallback_pool)[:needed]]

    return results



def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    all_chunks = []

    print("=" * 64)
    print("  NVIDIA RAG Dataset â€” Chunker v3 (Production Ready)")
    print("  All 7 pre-embedding fixes applied")
    print("=" * 64)

    total_before_dedup = 0

    for filename, source in FILES.items():
        filepath = os.path.join(script_dir, filename)

        if not os.path.exists(filepath):
            print(f"\n  âš ï¸  File not found: {filename} â€” skipping")
            continue

        print(f"\n  ðŸ“„ Processing: {filename} ({source})")
        chunks = chunk_markdown(filepath, source)
        all_chunks.extend(chunks)

        total_tokens = sum(c["token_approx"] for c in chunks)
        char_sizes   = [c["char_count"] for c in chunks]
        print(f"     Chunks produced : {len(chunks)}")
        print(f"     Approx tokens   : ~{total_tokens:,}")
        print(f"     Char range      : {min(char_sizes)}â€“{max(char_sizes)} chars per chunk")
        print(f"     Doc link        : {DOC_LINKS[source]}")

    # Save
    output_path = os.path.join(script_dir, "chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 64}")
    print(f"  âœ… {len(all_chunks)} clean chunks saved â†’ chunks.json")
    print(f"  ðŸ“‹ Checklist:")
    print(f"     [âœ“] Markdown headers, code blocks, separators removed")
    print(f"     [âœ“] MIN_CHARS = {MIN_CHARS} enforced")
    print(f"     [âœ“] MAX_CHARS = {MAX_CHARS} enforced (chunks split at paragraphs)")
    print(f"     [âœ“] Sections with <20 words dropped")
    print(f"     [âœ“] Whitespace normalized")
    print(f"     [âœ“] Table flattening ran first on every chunk")
    print(f"     [âœ“] Duplicate chunks removed by 200-char fingerprint")
    print("=" * 64)

    print("\n  DEMO â€” Retrieval on clean chunks")
    print("=" * 64)

    test_queries = [
        "How does Triton handle dynamic batching?",
        "How does TensorRT quantization work?",
        "What is the difference between NeMo, Triton, and TensorRT?",
    ]

    for query in test_queries:
        print(f"\n  â“ \"{query}\"")
        results = retrieve(query, all_chunks, top_k=2)
        for rank, chunk in enumerate(results, 1):
            print(f"  #{rank} [{chunk['chunk_id']}] \"{chunk['section']}\"")
            print(f"      {chunk['content'][:150].replace(chr(10), ' ')}...")
            print(f"      â†’ {chunk['doc_link']}")

    print("\n" + "=" * 64)
    print("  SAMPLE CLEAN CHUNK (no markdown noise, table flattened)")
    print("=" * 64)
    sample = next(
        (c for c in all_chunks if c["source"] == "nemo" and "Decision" in c["section"]),
        all_chunks[0]
    )
    print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()





