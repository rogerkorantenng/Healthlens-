# app.py — HealthLens (Two Tabs: Upload & Ask)
# --------------------------------------------
# One-file Gradio app: ingest PDFs/CSVs/TXTs -> Vertex AI embeddings -> Elastic index
# Ask questions -> hybrid search (BM25 + kNN/script_score) -> Gemini answer with citations.
#
# .env keys (fill with real values):
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
# pip install gradio python-dotenv elasticsearch pypdf pandas requests google-cloud-aiplatform

import os, re, json, uuid
from typing import List, Dict, Tuple
import pandas as pd
import requests

# quieten gRPC chatter locally
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
    raise RuntimeError("Set VERTEX_PROJECT_ID / VERTEX_LOCATION / GOOGLE_APPLICATION_CREDENTIALS in .env")
if not (ES_ENDPOINT and ES_API_KEY):
    raise RuntimeError("Set ELASTIC_CLOUD_ENDPOINT / ELASTIC_API_KEY in .env")
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
            "source":   {"type": "keyword"},
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
# 2) Ingest helpers
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

def chunk_text(text: str, size=1200, overlap=200) -> List[str]:
    chunks, n, i = [], len(text), 0
    while i < n:
        end = min(n, i + size)
        chunk = text[i:end]
        cut = chunk.rfind(".")
        if cut > int(size * 0.6):
            end = i + cut + 1
            chunk = text[i:end]
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        i = max(end - overlap, end)
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
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        raise RuntimeError("File produced no chunks (empty text or parse issue).")
    return index_chunks(title, path, chunks, meta)

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
        # Fallback if top-level knn is not allowed on your cluster
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

def answer_query(user_query: str, top_k: int, filters: Dict = None) -> Tuple[str, str, str]:
    ctx, max_score = hybrid_search(user_query, top_k, filters)
    if not ctx:
        return "No relevant context found. Please upload more reports.", "_No sources._", "_No snippets._"

    # Build context block for LLM
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
    resp = llm.generate_content(prompt, generation_config=generation_config)
    answer_md = (resp.text or "")

    # Sources list
    seen, sources = set(), []
    for c in ctx:
        key = (c["title"], c["source"])
        if key not in seen:
            seen.add(key)
            sources.append(f"- **{c['title']}** — `{c['source']}`")
    sources_md = "\n".join(sources) if sources else "_No sources._"

    # Snippet previews
    def hi(txt: str, q: str) -> str:
        terms = [t for t in re.findall(r"\w+", q.lower()) if len(t) > 3]
        out = txt
        for t in set(terms):
            out = re.sub(fr"(?i)\b({re.escape(t)})\b", r"**\1**", out)
        return out
    parts = []
    for c in ctx[:3]:
        snippet = c["text"]
        snippet = (snippet[:600] + "…") if len(snippet) > 600 else snippet
        parts.append(f"**{c['title']}**  \n{hi(snippet, user_query)}  \n`{c['source']}`")
    snippets_md = "\n\n---\n\n".join(parts) if parts else "_No snippets._"

    # add a light confidence hint (heuristic)
    conf = "high" if max_score >= 2.0 else ("medium" if max_score >= 1.0 else "low")
    answer_md = f"**Confidence:** {conf}\n\n" + answer_md

    return answer_md, f"### Sources\n{sources_md}", f"### Top Snippets\n{snippets_md}"

# --------------
# 4) Gradio UI (TWO TABS)
# --------------
THEME = gr.themes.Soft(primary_hue="blue")
CSS = """
footer {visibility: hidden}
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

def ui_ask(question, region, month, top_k):
    if not question or not question.strip():
        return "Please type a question.", "", ""
    filters = {"region": region or None, "month": month or None}
    try:
        ans_md, sources_md, snippets_md = answer_query(question, int(top_k), filters)
        return ans_md, sources_md, snippets_md
    except Exception as e:
        return f"❌ Query error: {e}", "", ""

with gr.Blocks(title="HealthLens", theme=THEME, css=CSS) as demo:
    gr.Markdown("# 🌍 HealthLens — AI Health Data Search\nTwo tabs: **Upload** and **Ask**.")

    with gr.Tab("Upload"):
        gr.Markdown("Upload a PDF / CSV / TXT to index it for search.")
        file = gr.File(label="File")
        title = gr.Textbox(label="Title (optional)")
        region = gr.Textbox(label="Region (optional)", placeholder="e.g., Northern")
        month  = gr.Textbox(label="Month (optional, e.g., 2025-07)")
        out_u = gr.Markdown()
        gr.Button("Ingest", variant="primary").click(ui_ingest, [file, title, region, month], [out_u])

    with gr.Tab("Ask"):
        gr.Markdown("Ask a question. Optionally filter by region/month.")
        q = gr.Textbox(label="Your question", placeholder="e.g., When was the initial alert received?")
        topk = gr.Slider(3, 20, value=DEFAULT_TOP_K, step=1, label="Top-K passages")
        region_q = gr.Textbox(label="Region filter (optional)")
        month_q  = gr.Textbox(label="Month filter (optional, e.g., 2025-07)")
        out_a = gr.Markdown()
        out_src = gr.Markdown()
        out_snip = gr.Markdown()
        gr.Button("Ask", variant="primary").click(
            ui_ask, [q, region_q, month_q, topk], [out_a, out_src, out_snip]
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
