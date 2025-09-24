# app.py — HealthLens (Two Tabs + Streaming + Readable UI + Web Ingest Agent)
# ---------------------------------------------------------------------------
# Upload PDF/CSV/TXT -> Vertex AI embeddings -> Elasticsearch (serverless-safe)
# Paste URLs -> Agent fetches pages (robots-aware) -> cleans text -> chunks -> embeds -> indexes
# Ask -> hybrid search (BM25 + kNN/script_score) -> Gemini **streamed** answers
#
# .env (use forward slashes on Windows paths):
#   GOOGLE_APPLICATION_CREDENTIALS=C:/path/to/service-account.json
#   VERTEX_PROJECT_ID=your-project-id
#   VERTEX_LOCATION=us-central1
#   VERTEX_MODEL_CHAT=gemini-1.5-flash-001
#   VERTEX_MODEL_EMBED=text-embedding-004
#   ELASTIC_CLOUD_ENDPOINT=https://<your-id>.<region>.elastic.cloud:443
#   ELASTIC_API_KEY=<api_key>
#   ELASTIC_INDEX=healthlens-docs
#   CHUNK_SIZE=1200
#   CHUNK_OVERLAP=200
#   TOP_K=8
#
# Install:
#   pip install gradio python-dotenv elasticsearch pypdf pandas google-cloud-aiplatform trafilatura

import os, re, json, uuid, time
from typing import List, Dict, Tuple
from urllib.parse import urlsplit
import urllib.robotparser as urobot
import pandas as pd

# Quiet local gRPC warnings (optional)
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

import gradio as gr
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
from pypdf import PdfReader

# Vertex AI
import vertexai
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel

# Web extraction
import trafilatura

# ----------------------------
# 0) Config & Initialization
# ----------------------------
load_dotenv()

PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", "")
LOCATION   = os.getenv("VERTEX_LOCATION", "us-central1")
KEY_PATH   = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
CHAT_MODEL = os.getenv("VERTEX_MODEL_CHAT", "gemini-1.5-flash-001")
EMB_MODEL  = os.getenv("VERTEX_MODEL_EMBED", "text-embedding-004")

ES_ENDPOINT = os.getenv("ELASTIC_CLOUD_ENDPOINT", "")
ES_API_KEY  = os.getenv("ELASTIC_API_KEY", "")
INDEX_NAME  = os.getenv("ELASTIC_INDEX", "healthlens-docs").strip()

CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", "200"))
DEFAULT_TOP_K  = int(os.getenv("TOP_K", "8"))

if not (PROJECT_ID and LOCATION and KEY_PATH):
    raise RuntimeError("Missing Vertex config (.env): VERTEX_PROJECT_ID / VERTEX_LOCATION / GOOGLE_APPLICATION_CREDENTIALS")
if not (ES_ENDPOINT and ES_API_KEY):
    raise RuntimeError("Missing Elastic config (.env): ELASTIC_CLOUD_ENDPOINT / ELASTIC_API_KEY")
if any(c in INDEX_NAME for c in ' "*,/<>?\\|'):
    raise ValueError(f"Invalid ELASTIC_INDEX: {INDEX_NAME}")

# Vertex init (explicit creds = reliable on Windows)
creds = service_account.Credentials.from_service_account_file(KEY_PATH)
vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=creds, api_transport="rest")
llm = GenerativeModel(CHAT_MODEL)
embedder = TextEmbeddingModel.from_pretrained(EMB_MODEL)

# Elastic client
es = Elasticsearch(ES_ENDPOINT, api_key=ES_API_KEY)

# ---------------------------------
# 1) Ensure index exists (serverless-safe)
# ---------------------------------
INDEX_SCHEMA = {
    "mappings": {
        "properties": {
            "doc_id":   {"type": "keyword"},
            "source":   {"type": "keyword"},   # file path or URL
            "title":    {"type": "text"},
            "text":     {"type": "text"},
            "metadata": {"type": "object"},
            "embedding": {"type": "dense_vector", "dims": 768, "index": True, "similarity": "cosine"},
        }
    }
}
def ensure_index():
    try:
        if not es.indices.exists(index=INDEX_NAME):
            es.indices.create(index=INDEX_NAME, body=INDEX_SCHEMA)
    except Exception as e:
        raise RuntimeError(f"Error ensuring index: {e}")
ensure_index()

# -----------------------
# 2) Ingest helpers (files)
# -----------------------
def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join([(p.extract_text() or "") for p in reader.pages])

def read_csv(path: str) -> str:
    df = pd.read_csv(path)
    preview = df.head(20).to_markdown(index=False)
    return f"CSV_TABLE_PREVIEW:\n{preview}\n\nRAW_CSV_PATH:{path}"

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

def sentence_chunks(text: str, size=1200, overlap=200) -> List[str]:
    # sentence-aware chunking for cleaner context
    sents = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) + 1 <= size:
            cur = (cur + " " + s).strip()
        else:
            if cur: chunks.append(cur)
            cur = s
    if cur: chunks.append(cur)
    if overlap > 0 and chunks:
        fused = []
        for i, ch in enumerate(chunks):
            if i > 0:
                prev_tail = chunks[i-1][-overlap:]
                ch = (prev_tail + " " + ch).strip()
            fused.append(ch)
        chunks = fused
    return chunks

def embed(texts: List[str]) -> List[List[float]]:
    return [e.values for e in embedder.get_embeddings(texts)]

def index_chunks(title: str, source: str, chunks: List[str], metadata: Dict):
    vectors = embed(chunks)
    doc_uuid = str(uuid.uuid4())
    actions = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        actions.append({
            "_op_type": "index",
            "_index": INDEX_NAME,
            "_id": f"{doc_uuid}-{i}",
            "_source": {
                "doc_id": doc_uuid,
                "source": source,
                "title": title,
                "text": chunk,
                "metadata": metadata or {},
                "embedding": vec,
            },
        })
    helpers.bulk(es, actions)
    return doc_uuid

def ingest_path(path: str, title: str = None, meta: Dict = None) -> str:
    ext = os.path.splitext(path)[1].lower()
    title = title or os.path.basename(path)
    meta = meta or {}

    if ext == ".pdf":
        raw = read_pdf(path)
    elif ext == ".csv":
        raw = read_csv(path)
    elif ext in (".txt", ".md"):
        raw = read_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    text = clean_text(raw)
    chunks = sentence_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        raise RuntimeError("File produced no chunks (empty text or parse issue).")
    return index_chunks(title, path, chunks, meta)

# -----------------------
# 2b) Web Ingest Agent (URLs)
# -----------------------
USER_AGENT = "HealthLensBot/0.1 (+contact: you@example.com)"
ROBOTS_CACHE: Dict[str, urobot.RobotFileParser] = {}

def robots_ok(url: str) -> bool:
    """Respect robots.txt for the given URL (best-effort; fail-open if missing)."""
    try:
        parts = urlsplit(url)
        if not parts.scheme or not parts.netloc:
            return True
        robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
        rp = ROBOTS_CACHE.get(robots_url)
        if rp is None:
            rp = urobot.RobotFileParser()
            rp.set_url(robots_url)
            try:
                rp.read()
            except Exception:
                # Couldn't read robots; many sites omit it. Allow by default.
                ROBOTS_CACHE[robots_url] = rp
                return True
            ROBOTS_CACHE[robots_url] = rp
        # If parser has no entries, allow; else check.
        try:
            return rp.can_fetch(USER_AGENT, url)
        except Exception:
            return True
    except Exception:
        return True

def fetch_clean(url: str) -> str:
    """Fetch a URL and extract main text with trafilatura."""
    downloaded = trafilatura.fetch_url(url, no_ssl=True)
    if not downloaded:
        return ""
    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        include_links=False,
        favor_recall=True,
        include_formatting=False,
        url=url,
    )
    return (text or "").strip()

def ingest_url(url: str, meta: Dict) -> str:
    if not robots_ok(url):
        raise RuntimeError(f"Blocked by robots.txt: {url}")
    content = fetch_clean(url)
    if not content:
        raise RuntimeError(f"No extractable text at {url}")
    title = url  # keep simple; could parse <title> if desired
    chunks = sentence_chunks(clean_text(content), CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        raise RuntimeError(f"Page produced no chunks: {url}")
    return index_chunks(title, url, chunks, meta)

def ingest_urls(urls: List[str], meta: Dict) -> Dict[str, str]:
    """Return mapping {url: doc_id or 'ERROR: ...'}"""
    out = {}
    for u in urls:
        try:
            doc_id = ingest_url(u, meta)
            out[u] = doc_id
            time.sleep(0.4)  # gentle rate limit
        except Exception as e:
            out[u] = f"ERROR: {e}"
    return out

# --------------------
# 3) Retrieval + LLM
# --------------------
SYSTEM_PROMPT = (
    "You are HealthLens, a careful public-health assistant.\n"
    "Rules:\n"
    "1) Use ONLY the provided context snippets; do not invent facts.\n"
    "2) If evidence is missing, say: \"I don’t know from the provided reports.\" \n"
    "3) Cite after each bullet like [Title](Source).\n"
    "4) Keep it tight: 3–6 bullets + one 'Takeaway:' line.\n"
    "5) If a CSV preview appears, mention key rows/columns explicitly.\n"
)

def terms(field: str, size: int = 50) -> List[str]:
    agg_field = f"metadata.{field}.keyword"
    body = {"size": 0, "aggs": {"x": {"terms": {"field": agg_field, "size": size, "order": {"_key": "asc"}}}}}
    try:
        r = es.search(index=INDEX_NAME, body=body)
        return [b["key"] for b in r.get("aggregations", {}).get("x", {}).get("buckets", [])]
    except Exception:
        return []

def hybrid_search(query: str, top_k: int, filters: Dict = None) -> Tuple[List[Dict], float]:
    qv = embedder.get_embeddings([query])[0].values

    must = [{"multi_match": {"query": query, "fields": ["text^2", "title"]}}]
    if filters:
        for k, v in filters.items():
            if v:
                must.append({"term": {f"metadata.{k}.keyword": v}})

    body_knn = {
        "size": top_k,
        "knn": {"field": "embedding", "query_vector": qv, "k": top_k, "num_candidates": top_k * 5},
        "query": {"bool": {"must": must}},
    }
    try:
        res = es.search(index=INDEX_NAME, body=body_knn)
    except Exception:
        # Fallback if your cluster disallows top-level knn
        body_script = {
            "size": top_k,
            "query": {
                "script_score": {
                    "query": {"bool": {"must": must}},
                    "script": {"source": "cosineSimilarity(params.q, 'embedding') + 1.0", "params": {"q": qv}},
                }
            },
        }
        res = es.search(index=INDEX_NAME, body=body_script)

    hits = res.get("hits", {}).get("hits", [])
    max_score = max((h.get("_score", 0.0) for h in hits), default=0.0)

    ctx = []
    for h in hits:
        s = h.get("_source", {})
        ctx.append({
            "text": s.get("text", ""),
            "title": s.get("title", "Untitled"),
            "source": s.get("source", "unknown"),
            "metadata": s.get("metadata", {}),
            "score": h.get("_score", 0.0),
        })
    return ctx, float(max_score)

def make_clickable(src: str) -> str:
    s = (src or "").strip()
    if s.startswith("http://") or s.startswith("https://"):
        return f"[{s}]({s})"
    return f"`{s}`"

# ---------- Streaming generator with readable layout ----------
def build_sources_md(ctx: List[Dict]) -> str:
    seen, lines = set(), []
    for c in ctx:
        key = (c["title"], c["source"])
        if key not in seen:
            seen.add(key)
            lines.append(f"- **{c['title']}** — {make_clickable(c['source'])}")
    return "### Sources\n" + ("\n".join(lines) if lines else "_No sources._")

def bold_terms(text: str, query: str) -> str:
    terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 3]
    out = text
    for t in set(terms):
        out = re.sub(fr"(?i)\b({re.escape(t)})\b", r"**\1**", out)
    return out

def build_snippets_md(ctx: List[Dict], query: str, n=3) -> str:
    parts = []
    for c in ctx[:n]:
        snippet = c["text"]
        snippet = (snippet[:600] + "…") if len(snippet) > 600 else snippet
        parts.append(f"**{c['title']}**  \n{bold_terms(snippet, query)}  \n{make_clickable(c['source'])}")
    return "### Top Snippets\n" + ("\n\n---\n\n".join(parts) if parts else "_No snippets._")

def stream_answer(user_query: str, region: str, month: str, top_k: int, last_answer_state: str):
    try:
        top_k = int(top_k)
        filters = {"region": region or None, "month": month or None}
        ctx, max_score = hybrid_search(user_query, top_k, filters)

        if not ctx:
            yield ("No relevant context found. Please upload more reports.", "_No sources._", "_No snippets._", "")
            return

        sources_md  = build_sources_md(ctx)
        snippets_md = build_snippets_md(ctx, user_query, n=3)

        conf = "high" if max_score >= 2.0 else ("medium" if max_score >= 1.0 else "low")
        answer_prefix = f"### Answer  \n<span class='badge'>confidence: {conf}</span>\n\n"

        # Build context for the LLM
        ctx_block = ""
        for i, c in enumerate(ctx, 1):
            ctx_block += (
                f"\n[CTX {i}] Title: {c['title']}\n"
                f"Source: {c['source']}\n"
                f"Meta: {c.get('metadata', {})}\n"
                f"Text: {c['text']}\n"
            )

        generation_config = {"temperature": 0.2, "max_output_tokens": 800, "top_p": 0.9}
        prompt = (
            f"{SYSTEM_PROMPT}\n\nUser question: {user_query}\n\n"
            f"Context snippets:\n{ctx_block}\n\n"
            "Output:\n- 3–6 bullets\n- 'Takeaway:' line\n- Inline citations after each bullet\n"
        )

        # First yield: show header + empty body so right column renders immediately
        current = answer_prefix
        yield (current, sources_md, snippets_md, current)

        # Stream tokens
        for chunk in llm.generate_content(prompt, generation_config=generation_config, stream=True):
            token = getattr(chunk, "text", None)
            if token:
                current += token
                yield (current, sources_md, snippets_md, current)

    except Exception as e:
        yield (f"❌ Query error: {e}", "", "", "")

# Download helper
def download_answer_md(answer_text: str):
    if not answer_text:
        return None
    path = f"healthlens_answer_{int(time.time())}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(answer_text)
    return path

# --------------------
# 4) Gradio UI (Two Tabs, readable)
# --------------------
THEME = gr.themes.Soft(primary_hue="blue")
CSS = """
footer {visibility: hidden}
#answer_md * {font-size: 1.05rem; line-height: 1.75;}
#answer_md ul {margin-left: 1.15rem; margin-top: .25rem;}
#answer_md li {margin: .18rem 0;}
#answer_md a {color: #64748b; text-decoration: none;}
.badge {display:inline-block;padding:4px 10px;border-radius:999px;background:#0ea5e9;color:white;font-size:12px;margin-left:8px}
"""

def ui_ingest(file, title, region, month):
    if file is None:
        return "Please upload a PDF/CSV/TXT."
    meta = {}
    if region: meta["region"] = region
    if month:  meta["month"] = month
    try:
        doc_id = ingest_path(file.name, title or os.path.basename(file.name), meta)
        return f"✅ Ingested **{os.path.basename(file.name)}**  \n`doc_id={doc_id}`"
    except Exception as e:
        return f"❌ Ingest failed: {e}"

def ui_ingest_urls(urls_text, region, month):
    if not urls_text or not urls_text.strip():
        return "Please paste one or more URLs (one per line)."
    meta = {}
    if region: meta["region"] = region
    if month:  meta["month"] = month
    urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
    results = ingest_urls(urls, meta)
    bullets = []
    for u, r in results.items():
        if str(r).startswith("ERROR:"):
            bullets.append(f"- ❌ {u} — {r}")
        else:
            bullets.append(f"- ✅ {u} — `doc_id={r}`")
    return "\n".join(bullets) if bullets else "Nothing to ingest."

def terms_safe(field):
    try: return terms(field) or []
    except: return []

REGIONS = terms_safe("region")
MONTHS  = terms_safe("month")

with gr.Blocks(title="HealthLens", theme=THEME, css=CSS) as demo:
    gr.Markdown("# 🌍 HealthLens — AI Health Data Search")

    with gr.Tab("Upload"):
        gr.Markdown("**Upload a PDF / CSV / TXT** to index it for search.")
        file = gr.File(label="File")
        title = gr.Textbox(label="Title (optional)")
        region = gr.Dropdown(choices=REGIONS, label="Region (optional)", allow_custom_value=True)
        month  = gr.Dropdown(choices=MONTHS,  label="Month (optional, e.g., 2025-07)", allow_custom_value=True)
        out_u = gr.Markdown()
        gr.Button("Ingest File", variant="primary").click(ui_ingest, [file, title, region, month], [out_u])

        gr.Markdown("### Or ingest from the web")
        urls_box = gr.Textbox(lines=6, label="Web URLs (one per line)", placeholder="https://www.who.int/...\nhttps://www.cdc.gov/...")
        out_u_urls = gr.Markdown()
        gr.Button("Fetch & Ingest URLs", variant="secondary").click(ui_ingest_urls, [urls_box, region, month], [out_u_urls])

    with gr.Tab("Ask"):
        gr.Markdown("Ask a question. Optionally filter by region/month.")
        q = gr.Textbox(label="Your question", placeholder="e.g., When was the initial alert received?")
        topk = gr.Slider(3, 20, value=DEFAULT_TOP_K, step=1, label="Top-K passages")
        region_q = gr.Dropdown(choices=REGIONS, label="Region filter (optional)", allow_custom_value=True)
        month_q  = gr.Dropdown(choices=MONTHS,  label="Month filter (optional, e.g., 2025-07)", allow_custom_value=True)

        # Two-column readable layout
        with gr.Row():
            with gr.Column(scale=2):
                ans_md = gr.Markdown(elem_id="answer_md")
                with gr.Row():
                    last_answer = gr.State("")
                    dl = gr.DownloadButton("⬇️ Download answer (.md)", variant="secondary")
                    dl.click(download_answer_md, inputs=[last_answer], outputs=[])

            with gr.Column(scale=1):
                with gr.Accordion("Sources", open=True):
                    out_src = gr.Markdown()
                with gr.Accordion("Top Snippets", open=False):
                    out_snip = gr.Markdown()

        # STREAMING: function yields (answer, sources, snippets, state)
        gr.Button("Ask (Streaming)", variant="primary").click(
            fn=stream_answer,
            inputs=[q, region_q, month_q, topk, last_answer],
            outputs=[ans_md, out_src, out_snip, last_answer],
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
