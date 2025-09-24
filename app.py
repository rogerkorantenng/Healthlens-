# app.py — HealthLens (Streaming + Sticky Ask + Web Ingest + Shallow Crawler + Live Crawl Progress)
# -----------------------------------------------------------------------------------------------
# Upload PDF/CSV/TXT or paste URLs -> Vertex AI embeddings -> Elasticsearch (serverless-safe)
# Crawl (shallow BFS) -> respects robots.txt -> ingest pages, with live progress streaming
# Ask -> hybrid search (BM25 + kNN/script_score) -> Gemini **streamed** answer with citations & snippets
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

import os, re, json, uuid, time
from typing import List, Dict, Tuple, Optional
from collections import deque
from urllib.parse import urlsplit, urljoin, urldefrag
import urllib.robotparser as urobot
import pandas as pd

# Quiet local gRPC warnings (optional)
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

import gradio as gr
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
from pypdf import PdfReader
from bs4 import BeautifulSoup

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
                ROBOTS_CACHE[robots_url] = rp
                return True
            ROBOTS_CACHE[robots_url] = rp
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
    title = url
    chunks = sentence_chunks(clean_text(content), CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        raise RuntimeError(f"Page produced no chunks: {url}")
    return index_chunks(title, url, chunks, meta)

def ingest_urls(urls: List[str], meta: Dict) -> Dict[str, str]:
    out = {}
    for u in urls:
        try:
            doc_id = ingest_url(u, meta)
            out[u] = doc_id
            time.sleep(0.4)
        except Exception as e:
            out[u] = f"ERROR: {e}"
    return out

# --------- UI wrapper for Web URLs (MISSING BEFORE — now added) ----------
def ui_ingest_urls(urls_text, region, month):
    """Web ingest UI wrapper: parse textarea -> call ingest_urls(url_list, meta)."""
    if not urls_text or not urls_text.strip():
        return "Please paste one or more URLs (one per line)."
    urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
    meta = {}
    if region: meta["region"] = region
    if month:  meta["month"]  = month
    results = ingest_urls(urls, meta)
    lines = []
    for u, r in results.items():
        if str(r).startswith("ERROR:"):
            lines.append(f"- ❌ {u} — {r}")
        else:
            lines.append(f"- ✅ {u} — `doc_id={r}`")
    return "\n".join(lines) if lines else "Nothing to ingest."

# -----------------------
# Shallow Crawler helpers
# -----------------------
def _normalize_url(href: str, base: str) -> Optional[str]:
    """Make absolute, drop fragments, keep http(s) only."""
    if not href:
        return None
    absu = urljoin(base, href.strip())
    absu, _ = urldefrag(absu)  # remove #fragment
    parts = urlsplit(absu)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None
    host = parts.netloc.lower()
    scheme = parts.scheme.lower()
    path = parts.path or "/"
    q = f"?{parts.query}" if parts.query else ""
    return f"{scheme}://{host}{path}{q}"

def _same_domain(u1: str, u2: str) -> bool:
    try:
        return urlsplit(u1).netloc.lower() == urlsplit(u2).netloc.lower()
    except Exception:
        return False

def _extract_links(html: str, base_url: str) -> List[str]:
    """Pull <a href>, absolutize, basic dedupe."""
    out = []
    try:
        soup = BeautifulSoup(html or "", "html.parser")
        for a in soup.find_all("a", href=True):
            u = _normalize_url(a["href"], base_url)
            if u:
                out.append(u)
    except Exception:
        pass
    seen, deduped = set(), []
    for u in out:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped

def crawl_site(seeds: List[str], same_domain_only: bool, max_pages: int, max_depth: int,
               allow_re: Optional[str] = None, deny_re: Optional[str] = None) -> List[str]:
    """
    BFS crawl. Returns list of URLs to ingest (visited pages).
    """
    allow_pat = re.compile(allow_re) if (allow_re or "").strip() else None
    deny_pat  = re.compile(deny_re)  if (deny_re  or "").strip() else None

    visited: set[str] = set()
    queue: deque[Tuple[str, int]] = deque()
    to_ingest: List[str] = []

    # seed queue
    for s in seeds:
        u = _normalize_url(s, s)
        if u:
            queue.append((u, 0))

    while queue and len(to_ingest) < max_pages:
        url, depth = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        if not robots_ok(url):
            continue

        # fetch raw HTML for link extraction
        html = trafilatura.fetch_url(url, no_ssl=True)
        if not html:
            continue

        # mark this page to ingest
        to_ingest.append(url)

        # expand links if depth allows
        if depth < max_depth:
            links = _extract_links(html, url)
            for link in links:
                if same_domain_only and not _same_domain(link, url):
                    continue
                if allow_pat and not allow_pat.search(link):
                    continue
                if deny_pat and deny_pat.search(link):
                    continue
                if link not in visited:
                    queue.append((link, depth + 1))

        time.sleep(0.3)
        if len(to_ingest) >= max_pages:
            break

    return to_ingest

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
# 3b) Streaming crawl UI handler (live progress)
# --------------------
def ui_crawl_stream(seeds_text, same_domain_only, max_pages, max_depth, allow_re, deny_re, region, month):
    """Stream crawl progress with a final full list of all crawled pages."""
    if not seeds_text or not seeds_text.strip():
        yield "Please enter at least one seed URL."
        return

    # Parse inputs
    seeds = [s.strip() for s in seeds_text.splitlines() if s.strip()]
    try:
        max_pages = int(max_pages)
        max_depth = int(max_depth)
    except Exception:
        yield "Max pages/depth must be numbers."
        return

    # Tagging
    meta = {}
    if region: meta["region"] = region
    if month:  meta["month"] = month

    # Compile filters
    allow_pat = re.compile(allow_re) if (allow_re or "").strip() else None
    deny_pat  = re.compile(deny_re)  if (deny_re  or "").strip() else None

    # BFS setup
    visited_set: set[str] = set()
    visited_list: list[str] = []  # keep order for final report
    queue: deque[tuple[str, int]] = deque()
    for s in seeds:
        u = _normalize_url(s, s)
        if u:
            queue.append((u, 0))

    crawled = 0
    ok_urls: list[str] = []
    err_list: list[tuple[str, str]] = []  # (url, error)

    log_lines = [f"⏳ Starting crawl with {len(seeds)} seed(s)…  ",
                 f"**Max pages:** {max_pages} • **Max depth:** {max_depth} • **Same domain:** {bool(same_domain_only)}"]
    yield "\n".join(log_lines)

    while queue and crawled < max_pages:
        url, depth = queue.popleft()
        if url in visited_set:
            continue
        visited_set.add(url)
        visited_list.append(url)

        # robots
        if not robots_ok(url):
            log_lines.append(f"- 🚫 Blocked by robots.txt: {url}")
            yield "\n".join(log_lines)
            continue

        # fetch raw HTML (for links)
        html = trafilatura.fetch_url(url, no_ssl=True)
        if not html:
            log_lines.append(f"- ⚠️ No HTML content: {url}")
            yield "\n".join(log_lines)
            continue

        # Ingest this page
        try:
            doc_id = ingest_url(url, meta)
            crawled += 1
            ok_urls.append(url)
            log_lines.append(f"{crawled}. ✅ Ingested (depth {depth}) — {url}  \n`doc_id={doc_id}`")
        except Exception as e:
            crawled += 1
            err_list.append((url, str(e)))
            log_lines.append(f"{crawled}. ❌ Failed (depth {depth}) — {url}  \n`{e}`")

        # Expand links if depth allows
        if depth < max_depth:
            for link in _extract_links(html, url):
                if same_domain_only and not _same_domain(link, url):
                    continue
                if allow_pat and not allow_pat.search(link):
                    continue
                if deny_pat and deny_pat.search(link):
                    continue
                if link not in visited_set:
                    queue.append((link, depth + 1))

        time.sleep(0.3)  # be polite
        yield "\n".join(log_lines)  # accumulate output so far

        if crawled >= max_pages:
            break

    # Final full report
    log_lines.append("")
    # log_lines.append(f"**Done.** Visited: {len(visited_list)} • Ingested OK: {len(ok_urls)} • Errors: {len(err_list)}")
    log_lines.append("")
    for i, u in enumerate(visited_list, 1):
        log_lines.append(f"{i}. {u}")

    if ok_urls:
        log_lines.append("")
        # log_lines.append(f"### Ingested OK ({len(ok_urls)})")
        for u in ok_urls:
            log_lines.append(f"- {u}")

    if err_list:
        log_lines.append("")
        # log_lines.append(f"### Errors ({len(err_list)})")
        for u, e in err_list:
            log_lines.append(f"- {u} — `{e}`")

    yield "\n".join(log_lines)


# --------------------
# 4) Gradio UI (Two Tabs, sticky ask, loading UX, Upload sub-tabs)
# --------------------
THEME = gr.themes.Soft(primary_hue="blue")
CSS = """
footer {visibility: hidden}
#answer_md * {font-size: 1.05rem; line-height: 1.75;}
#answer_md ul {margin-left: 1.15rem; margin-top: .25rem;}
#answer_md li {margin: .18rem 0;}
#answer_md a {color: #64748b; text-decoration: none;}
.badge {display:inline-block;padding:4px 10px;border-radius:999px;background:#0ea5e9;color:white;font-size:12px;margin-left:8px}
.status {font-size: .95rem; opacity: .8}

/* sticky ask bar */
#ask_bar {
  position: sticky;
  top: 0;
  z-index: 60;
  background: var(--background-fill-primary, #fff);
  border-bottom: 1px solid #e5e7eb;
  box-shadow: 0 2px 8px rgba(0,0,0,.04);
  padding: 10px 8px 12px;
}
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

def terms_safe(field):
    try: return terms(field) or []
    except: return []

REGIONS = terms_safe("region")
MONTHS  = terms_safe("month")

with gr.Blocks(title="HealthLens", theme=THEME, css=CSS) as demo:
    gr.Markdown("# 🌍 HealthLens — AI Health Data Search")

    # ---------------------------
    # Upload (with sub-tabs)
    # ---------------------------
    with gr.Tab("Upload"):
        gr.Markdown("Manage your corpus via file upload, direct URLs, or a shallow crawl.")

        with gr.Tabs():
            # Sub-tab 1: Upload a file
            with gr.TabItem("Upload a PDF / CSV / TXT"):
                file = gr.File(label="File")
                title = gr.Textbox(label="Title (optional)")
                region_file = gr.Dropdown(choices=REGIONS, label="Region (optional)", allow_custom_value=True)
                month_file  = gr.Dropdown(choices=MONTHS,  label="Month (optional, e.g., 2025-07)", allow_custom_value=True)

                status_file = gr.Markdown(elem_classes=["status"])
                out_file = gr.Markdown()
                btn_file = gr.Button("Ingest File", variant="primary")

                # Loading UX for file ingest
                btn_file.click(
                    lambda: (gr.update(value="Ingesting…", interactive=False), "⏳ Ingesting file…"),
                    inputs=[],
                    outputs=[btn_file, status_file],
                ).then(
                    lambda f, t, r, m: ui_ingest(f, t, r, m), [file, title, region_file, month_file], [out_file]
                ).then(
                    lambda: (gr.update(value="Ingest File", interactive=True), ""),
                    inputs=[],
                    outputs=[btn_file, status_file],
                )

            # Sub-tab 2: Ingest from the Web
            with gr.TabItem("Ingest from the Web"):
                urls_box = gr.Textbox(
                    lines=6,
                    label="Web URLs (one per line)",
                    placeholder="https://www.who.int/...\nhttps://www.cdc.gov/..."
                )
                region_web = gr.Dropdown(choices=REGIONS, label="Region (optional)", allow_custom_value=True)
                month_web  = gr.Dropdown(choices=MONTHS,  label="Month (optional, e.g., 2025-07)", allow_custom_value=True)

                status_web = gr.Markdown(elem_classes=["status"])
                out_web = gr.Markdown()
                btn_web = gr.Button("Fetch & Ingest URLs", variant="secondary")

                # Loading UX for URL ingest
                btn_web.click(
                    lambda: (gr.update(value="Fetching…", interactive=False), "⏳ Fetching pages…"),
                    inputs=[],
                    outputs=[btn_web, status_web],
                ).then(
                    ui_ingest_urls, [urls_box, region_web, month_web], [out_web]
                ).then(
                    lambda: (gr.update(value="Fetch & Ingest URLs", interactive=True), ""),
                    inputs=[],
                    outputs=[btn_web, status_web],
                )

            # Sub-tab 3: Crawl a Site (shallow, live progress)
            with gr.TabItem("Crawl a Site (shallow)"):
                crawl_seeds = gr.Textbox(
                    lines=4,
                    label="Seed URL(s) — one per line",
                    placeholder="https://www.who.int/emergencies/disease-outbreak-news\nhttps://www.cdc.gov/outbreaks/",
                )
                with gr.Row():
                    crawl_same = gr.Checkbox(value=True, label="Limit to same domain")
                    crawl_max_pages = gr.Slider(5, 200, value=30, step=1, label="Max pages")
                    crawl_max_depth = gr.Slider(0, 3, value=1, step=1, label="Max depth (clicks)")
                with gr.Row():
                    crawl_allow = gr.Textbox(label="Allow (regex, optional)", placeholder=r"^https?://([a-z0-9-]+\.)?example\.com/")
                    crawl_deny  = gr.Textbox(label="Deny (regex, optional)",  placeholder=r"\.pdf$|/amp/")

                region_crawl = gr.Dropdown(choices=REGIONS, label="Region (optional)", allow_custom_value=True)
                month_crawl  = gr.Dropdown(choices=MONTHS,  label="Month (optional, e.g., 2025-07)", allow_custom_value=True)

                status_crawl = gr.Markdown(elem_classes=["status"])
                out_crawl = gr.Markdown()
                btn_crawl = gr.Button("Crawl & Ingest", variant="secondary")

                # Loading UX + streaming progress + auto-scroll to output
                btn_crawl.click(
                    lambda: (gr.update(value="Crawling…", interactive=False), "⏳ Crawling site(s)…"),
                    inputs=[],
                    outputs=[btn_crawl, status_crawl],
                    scroll_to_output=True,
                ).then(
                    ui_crawl_stream,
                    inputs=[crawl_seeds, crawl_same, crawl_max_pages, crawl_max_depth, crawl_allow, crawl_deny, region_crawl, month_crawl],
                    outputs=[out_crawl],
                ).then(
                    lambda: (gr.update(value="Crawl & Ingest", interactive=True), ""),
                    inputs=[],
                    outputs=[btn_crawl, status_crawl],
                )

    # ---------------------------
    # Ask (sticky bar + streaming)
    # ---------------------------
    with gr.Tab("Ask"):
        with gr.Column(elem_id="ask_bar"):
            gr.Markdown("Ask a question. Optionally filter by region/month.")
            with gr.Row():
                q = gr.Textbox(label="Your question", placeholder="e.g., When was the initial alert received?", scale=8)
                ask_btn = gr.Button("Ask (Streaming)", variant="primary", scale=2)
            with gr.Row():
                region_q = gr.Dropdown(choices=REGIONS, label="Region (optional)", allow_custom_value=True, scale=3)
                month_q  = gr.Dropdown(choices=MONTHS,  label="Month (optional, e.g., 2025-07)", allow_custom_value=True, scale=3)
                topk     = gr.Slider(3, 20, value=DEFAULT_TOP_K, step=1, label="Top-K", scale=4)
            ask_status = gr.Markdown(elem_classes=["status"])

        # Content below (scrolls under the sticky bar)
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

        # Loading UX for Ask: disable + label + status; then stream; then reset
        ask_btn.click(
            lambda: (gr.update(value="Thinking…", interactive=False), "⏳ Generating…"),
            inputs=[],
            outputs=[ask_btn, ask_status],
            scroll_to_output=True,  # auto-scrolls to the first output (ans_md)
        ).then(
            fn=stream_answer,
            inputs=[q, region_q, month_q, topk, last_answer],
            outputs=[ans_md, out_src, out_snip, last_answer],
        ).then(
            lambda: (gr.update(value="Ask (Streaming)", interactive=True), ""),
            inputs=[],
            outputs=[ask_btn, ask_status],
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
