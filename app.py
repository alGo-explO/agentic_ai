import json
import os
import re
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


load_dotenv()


APP_TITLE = "Agentic AI Assessment Design Assistant (Milestone 2)"
PERSIST_DIR = os.path.join(os.path.dirname(__file__), ".chroma")
COLLECTION_NAME = "pedagogy_guidelines"
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"

# Groq model IDs change over time; keep this list to "Production Models" from Groq docs.
# Source: https://console.groq.com/docs/models
GROQ_PRODUCTION_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "groq/compound",
    "groq/compound-mini",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]
GROQ_FALLBACK_MODEL = "llama-3.3-70b-versatile"


class HashEmbeddingFunction:
    """
    Offline-safe embedding fallback.

    This avoids runtime failures when the environment cannot download
    SentenceTransformer weights (e.g., restricted networks).
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def __call__(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for t in texts:
            # simple token hashing into a fixed-size bag; normalize to unit-ish length
            vec = [0.0] * self.dim
            tokens = re.findall(r"[A-Za-z0-9_]+", (t or "").lower())
            for tok in tokens:
                h = hashlib.sha256(tok.encode("utf-8")).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                sign = -1.0 if (h[4] & 1) else 1.0
                vec[idx] += sign
            norm = sum(v * v for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]
            out.append(vec)
        return out


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _extract_first_json_object(text: str) -> Optional[dict]:
    """
    Best-effort extraction of the first top-level JSON object from a string that may
    contain extra text/markdown around it.
    """
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    snippet = text[start : i + 1]
                    maybe = _safe_json_loads(snippet)
                    return maybe if isinstance(maybe, dict) else None
    return None


class AssessmentSummaryModel(BaseModel):
    # PDF: "Summary: Assessment Quality & Difficulty Dist."
    assessment_quality_and_difficulty_summary: str = ""


class RecommendationModel(BaseModel):
    # PDF functional requirement: "Generate structured improvement recommendations."
    recommendation: str = ""
    rationale: str = ""
    citations: List[str] = Field(default_factory=list)


class AssessmentReportModel(BaseModel):
    # Keep only fields present in the PDF excerpt + required recommendations output.
    summary: AssessmentSummaryModel = Field(default_factory=AssessmentSummaryModel)
    gaps: List[str] = Field(default_factory=list)  # "Gaps: Identified Learning Gaps."
    recommendations: List[RecommendationModel] = Field(default_factory=list)  # structured improvements
    refs: List[str] = Field(default_factory=list)  # "Refs: Pedagogical References."
    disclaimer: str = ""  # "Disclaimer: Educational/Ethical notices"


def _coerce_report(obj: Any, *, raw_fallback: Optional[str] = None) -> Dict[str, Any]:
    """
    Enforce the Milestone 2 'structured output' format.
    Returns a dict that always matches AssessmentReportModel schema.
    """
    try:
        report = AssessmentReportModel.model_validate(obj)
        out = report.model_dump()
        if raw_fallback:
            out["_raw"] = raw_fallback
        return out
    except ValidationError:
        minimal = AssessmentReportModel().model_dump()
        if raw_fallback:
            minimal["_raw"] = raw_fallback
        return minimal


def _repair_json_with_llm(model: str, raw_text: str, schema_hint: str) -> Optional[dict]:
    system = (
        "You are a JSON repair tool. Convert the input into a single valid JSON object ONLY. "
        "Do not include markdown fences or extra text. Ensure it matches the given schema."
    )
    user = f"Schema:\n{schema_hint}\n\nInput:\n{raw_text}".strip()
    fixed = _llm_chat(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=0.0)
    parsed = _safe_json_loads(fixed)
    if isinstance(parsed, dict):
        return parsed
    return _extract_first_json_object(fixed)


def _coerce_rows_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Accepts either JSON list-of-rows or a simple CSV-like format.
    Expected row fields (any subset):
      - item_id (str/int)
      - max_marks (number)
      - avg_score (number) OR correct_rate (0..1)
      - high_group_avg (number)
      - low_group_avg (number)
      - common_wrong (str)
    """
    text = (text or "").strip()
    if not text:
        return []

    maybe = _safe_json_loads(text)
    if isinstance(maybe, list):
        return [r for r in maybe if isinstance(r, dict)]

    # Very small CSV-like parser: header line required.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    header = [h.strip() for h in lines[0].split(",")]
    rows: List[Dict[str, Any]] = []
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != len(header):
            continue
        row: Dict[str, Any] = {}
        for k, v in zip(header, parts):
            if v == "":
                row[k] = None
                continue
            if re.fullmatch(r"-?\d+(\.\d+)?", v):
                row[k] = float(v)
            else:
                row[k] = v
        rows.append(row)
    return rows


def _analyze_performance(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Lightweight analytics for Milestone 2: summary stats + heuristic flags.
    This intentionally avoids any Milestone 1 ML models.
    """
    if not rows:
        return {
            "has_data": False,
            "summary": "No performance data provided.",
            "difficulty_distribution": {},
            "flags": [],
            "per_item": [],
        }

    per_item = []
    diffs = []
    flags = []

    for r in rows:
        item_id = r.get("item_id", r.get("id", "item"))
        max_marks = r.get("max_marks")
        avg_score = r.get("avg_score")
        correct_rate = r.get("correct_rate")
        high = r.get("high_group_avg")
        low = r.get("low_group_avg")

        # difficulty proxy: correct_rate if available else avg/max
        p = None
        if isinstance(correct_rate, (int, float)):
            p = float(correct_rate)
        elif isinstance(avg_score, (int, float)) and isinstance(max_marks, (int, float)) and max_marks:
            p = float(avg_score) / float(max_marks)

        discrimination = None
        if isinstance(high, (int, float)) and isinstance(low, (int, float)):
            discrimination = float(high) - float(low)

        tag = None
        if p is not None:
            diffs.append(p)
            if p < 0.3:
                tag = "hard"
            elif p > 0.8:
                tag = "easy"
            else:
                tag = "medium"

        if discrimination is not None:
            if discrimination < 0.05:
                flags.append(
                    {
                        "item_id": item_id,
                        "type": "low_discrimination",
                        "detail": f"High-low difference is low ({discrimination:.2f}).",
                    }
                )

        if p is not None and (p < 0.2 or p > 0.9):
            flags.append(
                {
                    "item_id": item_id,
                    "type": "extreme_difficulty",
                    "detail": f"Estimated correctness is extreme ({p:.2f}).",
                }
            )

        per_item.append(
            {
                "item_id": item_id,
                "difficulty_proxy": p,
                "difficulty_tag": tag,
                "discrimination_proxy": discrimination,
                "common_wrong": r.get("common_wrong"),
            }
        )

    dist = {"easy": 0, "medium": 0, "hard": 0, "unknown": 0}
    for it in per_item:
        dist[it["difficulty_tag"] or "unknown"] += 1

    avg_p = sum(diffs) / len(diffs) if diffs else None
    summary = "Computed simple difficulty/discrimination proxies from the provided data."
    if avg_p is not None:
        summary += f" Mean estimated correctness across items: {avg_p:.2f}."

    # Basic "learning gap" hints from common_wrong
    gap_candidates = []
    for it in per_item:
        cw = it.get("common_wrong")
        if isinstance(cw, str) and cw.strip():
            gap_candidates.append({"item_id": it["item_id"], "signal": cw.strip()})

    return {
        "has_data": True,
        "summary": summary,
        "difficulty_distribution": dist,
        "flags": flags,
        "learning_gap_signals": gap_candidates[:10],
        "per_item": per_item,
    }


def _get_chroma_collection() -> Any:
    client = chromadb.PersistentClient(
        path=PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=DEFAULT_EMBED_MODEL
        )
    except Exception:
        embed_fn = HashEmbeddingFunction()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def _seed_if_empty(collection: Any) -> Tuple[bool, int]:
    try:
        existing = collection.count()
    except Exception:
        existing = 0
    if existing and existing > 0:
        return False, existing

    seed_path = os.path.join(os.path.dirname(__file__), "rag_seed", "pedagogy_guidelines.md")
    with open(seed_path, "r", encoding="utf-8") as f:
        text = f.read()

    # chunk by headings / paragraphs
    chunks = []
    buf = []
    for ln in text.splitlines():
        if ln.startswith("## ") and buf:
            chunks.append("\n".join(buf).strip())
            buf = [ln]
        else:
            buf.append(ln)
    if buf:
        chunks.append("\n".join(buf).strip())

    ids = [f"seed-{i}" for i in range(len(chunks))]
    metadatas = [{"source": "seed_corpus", "chunk": i} for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, metadatas=metadatas)
    return True, len(chunks)


def _add_user_docs(collection: Any, name: str, raw_text: str) -> int:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return 0
    # simple paragraph chunking
    paras = [p.strip() for p in re.split(r"\n{2,}", raw_text) if p.strip()]
    docs = []
    for p in paras:
        if len(p) < 40:
            continue
        docs.append(p)
    if not docs:
        return 0
    base = f"user-{re.sub(r'[^a-zA-Z0-9_-]+', '-', name)[:40]}-{int(datetime.utcnow().timestamp())}"
    ids = [f"{base}-{i}" for i in range(len(docs))]
    metadatas = [{"source": "user_upload", "name": name, "chunk": i} for i in range(len(docs))]
    collection.add(ids=ids, documents=docs, metadatas=metadatas)
    return len(docs)


def _retrieve(collection: Any, query: str, k: int = 6) -> List[Dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []
    res = collection.query(query_texts=[q], n_results=k, include=["documents", "metadatas", "distances"])
    out = []
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        out.append({"text": doc, "meta": meta or {}, "distance": dist})
    return out


def _groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY. Set it in Streamlit Secrets or environment.")
    return Groq(api_key=api_key)


def _llm_chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    client = _groq_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""
    except Exception:
        # Most common runtime issue here is a deprecated/invalid model ID.
        resp = client.chat.completions.create(
            model=GROQ_FALLBACK_MODEL,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


def _format_sources(retrieved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    refs = []
    for i, r in enumerate(retrieved, start=1):
        meta = r.get("meta") or {}
        source = meta.get("source", "unknown")
        name = meta.get("name")
        chunk = meta.get("chunk")
        label = f"[R{i}] {source}"
        if name:
            label += f" ({name})"
        if chunk is not None:
            label += f" chunk={chunk}"
        refs.append({"ref": f"R{i}", "label": label, "distance": r.get("distance"), "text": r.get("text", "")})
    return refs


@dataclass
class AgentState:
    query: str = ""
    course_context: str = ""
    performance_rows: List[Dict[str, Any]] = field(default_factory=list)
    performance_analysis: Dict[str, Any] = field(default_factory=dict)
    retrieved: List[Dict[str, Any]] = field(default_factory=list)
    source_refs: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: Dict[str, Any] = field(default_factory=dict)
    grounding_check: Dict[str, Any] = field(default_factory=dict)
    model: str = GROQ_FALLBACK_MODEL
    run_meta: Dict[str, Any] = field(default_factory=dict)


def node_analyze(state: AgentState) -> AgentState:
    state.performance_analysis = _analyze_performance(state.performance_rows)
    state.run_meta["analyzed_at"] = _now_iso()
    return state


def node_retrieve(state: AgentState) -> AgentState:
    collection = _get_chroma_collection()
    _seed_if_empty(collection)
    retrieval_query = state.query
    if state.course_context.strip():
        retrieval_query = f"{state.course_context.strip()}\n\n{state.query.strip()}"
    state.retrieved = _retrieve(collection, retrieval_query, k=8)
    state.source_refs = _format_sources(state.retrieved)
    state.run_meta["retrieved_at"] = _now_iso()
    return state


def node_recommend(state: AgentState) -> AgentState:
    refs_brief = "\n".join(
        [f"{r['ref']}: {r['label']}\n{r['text']}" for r in state.source_refs[:8]]
    )
    perf = state.performance_analysis or {}
    perf_json = json.dumps(perf, ensure_ascii=False)

    system = (
        "You are an agentic assessment design assistant for higher education. "
        "You must be conservative: only claim pedagogy guidance that is supported by the provided retrieved references. "
        "If evidence is insufficient, say so and ask for more context in the 'Open Questions' section. "
        "Return STRICT JSON only, matching the requested schema."
    )

    user = f"""
Assessment design query:
{state.query}

Course / context (optional):
{state.course_context}

Performance patterns (derived from user data; may be empty):
{perf_json}

Retrieved pedagogy references (cite as R1, R2...):
{refs_brief}

Produce a structured assessment report as JSON with this schema
(keep only these fields; do not add extra keys):
{{
  "summary": {{
    "assessment_quality_and_difficulty_summary": "string"
  }},
  "gaps": ["string", ...],
  "recommendations": [
    {{
      "recommendation": "string",
      "rationale": "string",
      "citations": ["R1", "R2"]
    }}
  ],
  "refs": ["R1", "R2"],
  "disclaimer": "string"
}}

Rules:
- Include citations for every advice item; if you cannot support it, omit it.
- Ensure the disclaimer mentions educational use, potential bias, and the need for instructor review.
""".strip()

    raw = _llm_chat(
        model=state.model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )

    parsed = _safe_json_loads(raw)
    if not isinstance(parsed, dict):
        parsed = _extract_first_json_object(raw)

    schema_hint = (
        "{"
        '"summary":{"assessment_quality_and_difficulty_summary":"string"},'
        '"gaps":["string"],'
        '"recommendations":[{"recommendation":"string","rationale":"string","citations":["R1"]}],'
        '"refs":["R1"],'
        '"disclaimer":"string"'
        "}"
    )
    if not isinstance(parsed, dict):
        parsed = _repair_json_with_llm(state.model, raw, schema_hint)

    report = _coerce_report(parsed, raw_fallback=raw)
    # Ensure refs has at least something when we retrieved context.
    if not report.get("refs") and state.source_refs:
        report["refs"] = [r["ref"] for r in state.source_refs[:3]]
    if not report.get("disclaimer"):
        report["disclaimer"] = (
            "Educational assistant output only. May contain errors or bias; must be reviewed and adapted by the instructor."
        )
    state.recommendations = report
    state.run_meta["recommended_at"] = _now_iso()
    return state


def node_grounding_check(state: AgentState) -> AgentState:
    """
    Lightweight evaluation report (Milestone 2 requirement): checks citation presence and
    asks the LLM to self-audit for unsupported claims.
    """
    rec = state.recommendations or {}
    rec_json = json.dumps(rec, ensure_ascii=False)
    refs = "\n".join([f"{r['ref']}: {r['text']}" for r in state.source_refs[:8]])

    system = (
        "You are a strict evaluator. Identify unsupported claims relative to the provided references. "
        "Return STRICT JSON only."
    )
    user = f"""
References:
{refs}

Assistant JSON output to evaluate:
{rec_json}

Return JSON schema:
{{
  "citation_coverage": {{
    "recommendations_with_citations_pct": "number",
    "drafts_with_citations_pct": "number"
  }},
  "unsupported_claims": [
    {{
      "location": "string",
      "claim": "string",
      "reason": "string"
    }}
  ],
  "overall_groundedness": "low|medium|high",
  "notes": ["string", ...]
}}
""".strip()

    # deterministic-ish
    raw = _llm_chat(
        model=state.model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.1,
    )
    parsed = _safe_json_loads(raw)
    if not isinstance(parsed, dict):
        parsed = _extract_first_json_object(raw)

    eval_schema_hint = (
        "{"
        '"citation_coverage":{"recommendations_with_citations_pct":0,"drafts_with_citations_pct":0},'
        '"unsupported_claims":[{"location":"string","claim":"string","reason":"string"}],'
        '"overall_groundedness":"low","notes":["string"]'
        "}"
    )
    if not isinstance(parsed, dict):
        parsed = _repair_json_with_llm(state.model, raw, eval_schema_hint) or {}
    if not isinstance(parsed, dict):
        parsed = {}
    parsed.setdefault("_raw", raw)
    parsed.setdefault("citation_coverage", {})
    parsed.setdefault("unsupported_claims", [])
    parsed.setdefault("overall_groundedness", "low")
    parsed.setdefault("notes", [])

    # local coverage stats
    recs = rec.get("recommendations") if isinstance(rec, dict) else None
    def _pct_with_citations(items: Any) -> float:
        if not isinstance(items, list) or not items:
            return 0.0
        ok = 0
        for it in items:
            c = it.get("citations") if isinstance(it, dict) else None
            if isinstance(c, list) and len(c) > 0:
                ok += 1
        return round(100.0 * ok / len(items), 1)

    parsed.setdefault("citation_coverage", {})
    parsed["citation_coverage"]["recommendations_with_citations_pct"] = _pct_with_citations(recs)
    parsed.setdefault("notes", [])
    parsed["notes"].append("This evaluation is heuristic and should be reviewed by an instructor.")
    state.grounding_check = parsed
    state.run_meta["evaluated_at"] = _now_iso()
    return state


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("analyze", node_analyze)
    g.add_node("retrieve", node_retrieve)
    g.add_node("recommend", node_recommend)
    g.add_node("evaluate", node_grounding_check)
    g.set_entry_point("analyze")
    g.add_edge("analyze", "retrieve")
    g.add_edge("retrieve", "recommend")
    g.add_edge("recommend", "evaluate")
    g.add_edge("evaluate", END)
    return g.compile()


def _render_architecture_diagram():
    mermaid = r"""
flowchart TD
  UI[Streamlit UI] -->|Query + optional performance data| G[LangGraph Workflow]
  G --> A[Analyze patterns\n(simple stats/heuristics)]
  G --> R[RAG Retrieve pedagogy\nChroma + embeddings]
  R --> LLM[Groq LLM\nreason + draft improvements]
  LLM --> E[Grounding check\n(citations + self-audit)]
  E --> OUT[Structured report\nrecommendations + drafts + refs]
"""
    st.code(mermaid.strip(), language="mermaid")


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    st.caption("Milestone 2 only: agentic reasoning + RAG + structured improvement report (no Milestone 1 ML).")

    with st.sidebar:
        model = st.selectbox("Model", options=GROQ_PRODUCTION_MODELS, index=0)

    tab_run, tab_rag, tab_about = st.tabs(["Run", "RAG", "About"])

    with tab_run:
        query = st.text_area(
            "Assessment design query",
            placeholder="Example: Improve a short-answer question on backpropagation; students are mixing up chain rule and gradient signs.",
            height=140,
        )
        course_context = st.text_area(
            "Course context (optional)",
            placeholder="Course: Intro ML. Topic: backpropagation. Level: 2nd year. Allowed tools: no internet.",
            height=90,
        )

        run = st.button("Generate report", type="primary", use_container_width=True)

    with tab_rag:
        st.write(f"**Chroma storage**: `{PERSIST_DIR}`")
        c1, c2 = st.columns([0.45, 0.55], gap="large")
        with c1:
            if st.button("Initialize / seed guidelines corpus", use_container_width=True):
                col = _get_chroma_collection()
                seeded, n = _seed_if_empty(col)
                st.success(f"{'Seeded' if seeded else 'Already initialized'}. Chunks in collection: {n}")
        with c2:
            with st.expander("Add extra guidelines text to Chroma", expanded=True):
                upload_name = st.text_input("Document name", value="institution_guidelines")
                upload_text = st.text_area(
                    "Guidelines text",
                    placeholder="Paste your department's assessment policy, rubric standards, or item-writing guidelines here.",
                    height=160,
                )
                if st.button("Add to Chroma", use_container_width=True):
                    col = _get_chroma_collection()
                    _seed_if_empty(col)
                    added = _add_user_docs(col, upload_name, upload_text)
                    if added:
                        st.success(f"Added {added} chunk(s).")
                    else:
                        st.info("No chunks added (text too short or empty).")

    with tab_about:
        with st.expander("System architecture (diagram)", expanded=False):
            _render_architecture_diagram()
        with st.expander("Input–output specification (Milestone 2)", expanded=False):
            st.markdown(
                "- **Inputs**: assessment query; optional course context; optional item-level performance summary.\n"
                "- **Processing**: LangGraph workflow with explicit state; Chroma RAG retrieval; Groq LLM reasoning.\n"
                "- **Outputs**: structured report (recommendations + drafts + refs + disclaimer) and a grounding check."
            )
        st.caption("Tip: Keep the default 'Run' tab minimal; use 'RAG' to manage retrieval sources.")

    if run:
        if not query.strip():
            st.error("Please enter an assessment design query.")
            return

        rows: List[Dict[str, Any]] = []
        state = AgentState(
            query=query.strip(),
            course_context=course_context.strip(),
            performance_rows=rows,
            model=model,
            run_meta={"started_at": _now_iso()},
        )

        graph = build_graph()
        with st.spinner("Running agentic workflow (analyze → retrieve → recommend → evaluate)..."):
            out_any = graph.invoke(state)

        # LangGraph commonly returns a plain dict state. Normalize for rendering + export.
        if isinstance(out_any, dict):
            out_state: Dict[str, Any] = out_any
        else:
            out_state = dict(getattr(out_any, "__dict__", {}))

        st.divider()
        st.subheader("Assessment report")

        rec_any = out_state.get("recommendations") or {}
        rec = _coerce_report(rec_any, raw_fallback=rec_any.get("_raw") if isinstance(rec_any, dict) else None)
        if isinstance(rec, dict) and rec.get("_raw"):
            with st.expander("Debug: raw model output", expanded=False):
                st.text(rec.get("_raw", ""))

        pu = rec.get("problem_understanding") or {}
        summ = rec.get("summary") or {}
        perf = out_state.get("performance_analysis") or {}

        c1, c2 = st.columns(2, gap="large")
        with c1:
            with st.container(border=True):
                st.markdown("### Assessment quality & difficulty")
                dist = perf.get("difficulty_distribution") or {}
                if dist:
                    st.markdown(
                        f"**Difficulty distribution**: easy={dist.get('easy',0)}, "
                        f"medium={dist.get('medium',0)}, hard={dist.get('hard',0)}, unknown={dist.get('unknown',0)}"
                    )
                st.markdown(
                    f"**Summary**: {(summ.get('assessment_quality_and_difficulty_summary') or perf.get('summary') or '—')}"
                )
                flags = perf.get("flags") or []
                if flags:
                    with st.expander("Flags from performance data", expanded=False):
                        for f in flags:
                            st.markdown(f"- **{f.get('item_id','item')}**: {f.get('type','flag')} — {f.get('detail','')}")

        with c2:
            with st.container(border=True):
                st.markdown("### Learning gaps")
                signals = perf.get("learning_gap_signals") or []
                if signals:
                    with st.expander("Signals from data", expanded=False):
                        for s in signals:
                            st.markdown(f"- **{s.get('item_id','item')}**: {s.get('signal','')}")
                gaps = rec.get("gaps") or []
                if gaps:
                    st.markdown("**Gaps (identified learning gaps)**")
                    for g in gaps:
                        st.markdown(f"- {g}")

        with st.container(border=True):
            st.markdown("### Recommendations (structured improvements)")
            recs = rec.get("recommendations") or []
            if not recs:
                st.write("—")
            for i, r in enumerate(recs, start=1):
                title = (r.get("recommendation") or f"Recommendation {i}").strip()
                cites = r.get("citations") or []
                cite_str = ", ".join(cites) if cites else "No citations"
                with st.expander(f"{i}. {title}  ({cite_str})", expanded=i <= 2):
                    st.markdown(f"**Rationale**: {r.get('rationale','—')}")

        with st.container(border=True):
            st.markdown("### References")
            refs_used = rec.get("refs") or []
            if refs_used:
                st.markdown(f"**Refs**: {', '.join(refs_used)}")
            with st.expander("Show retrieved reference chunks", expanded=False):
                st.json(out_state.get("source_refs") or [])

        with st.container(border=True):
            st.markdown("### Disclaimer")
            st.info(rec.get("disclaimer") or "Educational assistant output only; requires instructor review.")

        with st.expander("Evaluation (grounding report)", expanded=False):
            st.json(out_state.get("grounding_check") or {})

        with st.expander("Raw structured JSON (validated)", expanded=False):
            st.json(rec)

        st.subheader("Raw JSON export")
        st.download_button(
            "Download report JSON",
            data=json.dumps(
                {
                    "run_meta": out_state.get("run_meta") or {},
                    "inputs": {
                        "query": out_state.get("query", ""),
                        "course_context": out_state.get("course_context", ""),
                        "performance_rows": out_state.get("performance_rows") or [],
                    },
                    "performance_analysis": out_state.get("performance_analysis") or {},
                    "report": rec,
                    "evaluation": out_state.get("grounding_check") or {},
                },
                ensure_ascii=False,
                indent=2,
            ),
            file_name="assessment_report.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()

