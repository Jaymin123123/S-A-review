import os
import secrets
import html
import re
from io import BytesIO

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, Form, UploadFile, File, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from openai import OpenAI
import docx

torch.set_num_threads(1)

# Optional PDF/OCR libraries (best-effort)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
try:
    from pdfminer_high_level import extract_text as pdfminer_extract_text
except Exception:
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract_text
    except Exception:
        pdfminer_extract_text = None
try:
    import PyPDF2
except Exception:
    PyPDF2 = None
try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None

# -----------------------------
# Configuration
# -----------------------------
USERNAME = os.getenv("APP_USERNAME", "admin")
PASSWORD = os.getenv("APP_PASSWORD", "change_me")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
CLS_MODEL_NAME = os.getenv("CLS_MODEL_NAME", "your-rem-classifier-model")

TOP_K = int(os.getenv("TOP_K", "20"))
LOW_MARGIN = float(os.getenv("LOW_MARGIN", "0.1"))
AGAINST_THRESHOLD = float(os.getenv("AGAINST_THRESHOLD", "0.01"))

FLIP_LABELS = os.getenv("FLIP_LABELS", "1").strip() not in {"0", "false", "False", "no", "No"}

AGAINST_LABEL = 0
FOR_LABEL = 1

# -----------------------------
# Models
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

emb_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
emb_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(device).eval()

cls_tokenizer = AutoTokenizer.from_pretrained(CLS_MODEL_NAME)
classifier_model = AutoModelForSequenceClassification.from_pretrained(CLS_MODEL_NAME).to(device).eval()
NUM_LABELS = classifier_model.config.num_labels

LABEL_MAP = {}
try:
    if getattr(classifier_model.config, "id2label", None):
        LABEL_MAP = {int(k): str(v).upper() for k, v in classifier_model.config.id2label.items()}
except Exception:
    LABEL_MAP = {}

FOR_INDEX, AGAINST_INDEX = 1, 0
if LABEL_MAP:
    for idx, label in LABEL_MAP.items():
        if "FOR" in label:
            FOR_INDEX = idx
        if "AGAINST" in label:
            AGAINST_INDEX = idx

print("Classifier num_labels:", NUM_LABELS)
print("Label map:", LABEL_MAP or "(none)")
print("Using FOR_INDEX:", FOR_INDEX, "AGAINST_INDEX:", AGAINST_INDEX)
print("FLIP_LABELS:", FLIP_LABELS)

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
print("OpenAI client ready:", bool(client))

# -----------------------------
# Data
# -----------------------------
DF_PATH = os.getenv("POLICY_CSV", "investor_rem_policies.csv")
df = pd.read_csv(DF_PATH)
investor_policies = dict(zip(df["Investor"], df["RemunerationPolicy"]))

# CSV mapping + name matching helpers
CSV_MAP = {
    "autotrader": os.getenv("AUTOTRADER_CSV", "autotrader_against_votes.csv"),
    "unilever": os.getenv("UNILEVER_CSV", "unilever_against_votes.csv"),
    "sainsbury": os.getenv("SAINSBURY_CSV", "sainsbury_against_votes.csv"),
    "leg": os.getenv("LEG_CSV", "leg_against_votes.csv"),
    "asr": os.getenv("ASR_CSV", "asr_against_votes.csv"),
    "euronext": os.getenv("EURONEXT_CSV", "euronext_against_votes.csv"),
}


def _tokenise_name(s: str) -> list[str]:
    """Lowercase, alphanumeric-only tokens for name matching."""
    return [t for t in re.findall(r"[A-Za-z0-9]+", str(s).lower()) if t]


def _prefix_key_from_tokens(tokens: list[str]) -> str:
    """First two tokens joined if available; otherwise first token; else empty."""
    if not tokens:
        return ""
    return " ".join(tokens[:2]) if len(tokens) >= 2 else tokens[0]


# Index of investor prefixes (first 1 and first 2 tokens) -> investor names
INVESTOR_PREFIX_INDEX: dict[str, set[str]] = {}
for inv_name in investor_policies.keys():
    toks = _tokenise_name(inv_name)
    keys = set()
    if toks:
        keys.add(toks[0])
        keys.add(_prefix_key_from_tokens(toks))
    for k in keys:
        if not k:
            continue
        INVESTOR_PREFIX_INDEX.setdefault(k, set()).add(inv_name)


def _pick_manager_col(df_csv: pd.DataFrame) -> str | None:
    """Try common column names that could contain the manager/investor name."""
    lower = {c.lower(): c for c in df_csv.columns}
    candidates = [
        "vote manager", "manager", "votemanager",
        "investor", "investor name", "account", "organisation", "organization",
        "firm", "holder", "fund", "fund name",
    ]
    for c in candidates:
        if c in lower:
            return lower[c]
    for c in df_csv.columns:
        if df_csv[c].dtype == object:
            return c
    return None


def _filter_against_rows(df_csv: pd.DataFrame) -> pd.DataFrame:
    """If a vote/decision column exists, keep only rows that look like AGAINST; otherwise return as-is."""
    lower = {c.lower(): c for c in df_csv.columns}
    vote_candidates = ["vote", "decision", "voteresult", "vote result", "resolution vote", "voted"]
    for c in vote_candidates:
        if c in lower:
            col = lower[c]
            ser = df_csv[col].astype(str).str.lower()
            mask = ser.str.contains("against")
            if mask.any():
                return df_csv[mask]
            break
    return df_csv


def load_company_against_investors_from_csv(csv_path: str) -> set[str]:
    """
    Load the CSV and map vote-manager names to our investor list
    using first-two-term (or first-term) prefix matching.
    Returns a set of investor display names from investor_policies.
    """
    matched: set[str] = set()
    try:
        df_csv = pd.read_csv(csv_path)
    except Exception:
        return matched

    df_csv = _filter_against_rows(df_csv)
    manager_col = _pick_manager_col(df_csv)
    if not manager_col:
        return matched

    for raw_name in df_csv[manager_col].dropna().astype(str).tolist():
        toks = _tokenise_name(raw_name)
        key = _prefix_key_from_tokens(toks)
        tried = []
        if key:
            tried.append(key)
        if toks:
            tried.append(toks[0])

        for k in tried:
            invs = INVESTOR_PREFIX_INDEX.get(k)
            if invs:
                matched.update(invs)
                break

    return matched


# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI()
sessions: dict[str, str] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def login_required(request: Request):
    token = request.cookies.get("session")
    if token and token in sessions:
        return sessions[token]
    return RedirectResponse(url="/login", status_code=302)


def escape_html(s: str) -> str:
    return html.escape(s).replace("\n", "<br>")


# -----------------------------
# File text extraction helpers
# -----------------------------
def extract_text_from_docx_bytes(data: bytes) -> str:
    document = docx.Document(BytesIO(data))
    paras = [p.text for p in document.paragraphs if p.text and p.text.strip()]
    for table in getattr(document, "tables", []):
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text and cell.text.strip()]
            if cells:
                paras.append("\t".join(cells))
    return "\n".join(paras)


def extract_text_from_pdf_bytes(data: bytes) -> str:
    if fitz is not None:
        try:
            text_parts = []
            with fitz.open(stream=data, filetype="pdf") as doc:
                for page in doc:
                    text_parts.append(page.get_text("text"))
            text = "\n".join(t for t in text_parts if t)
            if text and text.strip():
                return text
        except Exception:
            pass
    if pdfminer_extract_text is not None:
        try:
            txt = pdfminer_extract_text(BytesIO(data))
            if txt and txt.strip():
                return txt
        except Exception:
            pass
    if PyPDF2 is not None:
        try:
            reader = PyPDF2.PdfReader(BytesIO(data))
            out = []
            for page in reader.pages:
                out.append(page.extract_text() or "")
            txt = "\n".join(out)
            if txt and txt.strip():
                return txt
        except Exception:
            pass
    if pytesseract is not None and Image is not None and fitz is not None:
        try:
            out = []
            with fitz.open(stream=data, filetype="pdf") as doc:
                for page in doc:
                    pix = page.get_pixmap(dpi=200)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    out.append(pytesseract.image_to_string(img))
            return "\n".join(out)
        except Exception:
            pass
    raise RuntimeError("Unable to extract text from PDF. Install PyMuPDF or pdfminer.six for best results.")


# -----------------------------
# Embeddings
# -----------------------------
def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    masked = last_hidden_state * attention_mask.unsqueeze(-1)
    lengths = attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
    return masked.sum(dim=1) / lengths


@torch.no_grad()
def get_embeddings(texts, batch_size: int = 64, max_length: int = 512):
    if not isinstance(texts, (list, tuple)):
        texts = [texts]
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = emb_tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        ).to(device)
        outputs = emb_model(**enc)
        sent_emb = _mean_pool(outputs.last_hidden_state, enc["attention_mask"])
        sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)
        all_vecs.append(sent_emb.cpu())
    return torch.cat(all_vecs, dim=0).numpy()


def get_embedding(text: str):
    return get_embeddings([text])[0]


# -----------------------------
# Chunking
# -----------------------------
def chunk_text(text: str, max_tokens: int = 512, stride: int = 256, min_tokens: int = 16):
    original_max = getattr(emb_tokenizer, "model_max_length", 512)
    try:
        emb_tokenizer.model_max_length = 10**9
        ids = emb_tokenizer.encode(text, add_special_tokens=False, truncation=False)
    finally:
        emb_tokenizer.model_max_length = original_max

    chunks = []
    for start in range(0, len(ids), stride):
        window = ids[start:start + max_tokens]
        if len(window) < min_tokens:
            continue
        chunk = emb_tokenizer.decode(window, skip_special_tokens=True)
        chunks.append(chunk)
        if start + max_tokens >= len(ids):
            break
    return chunks


# -----------------------------
# Classifier
# -----------------------------
@torch.no_grad()
def predict_vote(policy: str, chunk: str, max_length: int = 512):
    p = cls_tokenizer(policy, truncation=True, max_length=max_length // 2, add_special_tokens=False)
    c = cls_tokenizer(chunk, truncation=True, max_length=max_length // 2, add_special_tokens=False)

    ids = cls_tokenizer.build_inputs_with_special_tokens(p["input_ids"], c["input_ids"])
    token_type_ids = cls_tokenizer.create_token_type_ids_from_sequences(p["input_ids"], c["input_ids"])
    if len(ids) > max_length:
        ids = ids[:max_length]
        token_type_ids = token_type_ids[:max_length]
    attention_mask = [1] * len(ids)

    inputs = {
        "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long, device=device),
        "token_type_ids": torch.tensor([token_type_ids], dtype=torch.long, device=device),
    }
    logits = classifier_model(**inputs).logits.squeeze(0)

    if NUM_LABELS == 1:
        prob_against = torch.sigmoid(logits).item()
        pred = AGAINST_LABEL if prob_against >= 0.5 else FOR_LABEL
    else:
        probs = torch.softmax(logits, dim=-1)
        prob_against = probs[AGAINST_INDEX].item()
        prob_for = probs[FOR_INDEX].item()
        pred = AGAINST_LABEL if prob_against >= prob_for else FOR_LABEL

    if FLIP_LABELS:
        pred = FOR_LABEL if pred == AGAINST_LABEL else AGAINST_LABEL
        prob_against = 1.0 - prob_against

    return pred, float(prob_against)


# -----------------------------
# Decision helpers
# -----------------------------
def weighted_decision(scored, sims):
    votes = np.array([v for _, v, _ in scored], dtype=float)
    probs = np.array([p for _, _, p in scored], dtype=float)
    weights = sims + 1e-8
    weights = weights / weights.sum()

    votes_against = (votes == AGAINST_LABEL).astype(float)

    weighted_frac_against = float((votes_against * weights).sum())
    weighted_mean_prob = float((probs * weights).sum())

    maj = AGAINST_LABEL if weighted_frac_against >= AGAINST_THRESHOLD else FOR_LABEL
    conf = abs(weighted_mean_prob - 0.5)
    return maj, conf, weighted_frac_against, weighted_mean_prob


# -----------------------------
# GPT Reason Helper
# -----------------------------
def get_gpt_reason(policy_text: str, chunks: list[str]):
    if client is None:
        return None
    formatted_chunks = "\n".join(f"- {c}" for c in chunks[:TOP_K])
    prompt = (
        "An investor policy states:\n\n" + policy_text + "\n\n"
        "The company has disclosed the following relevant information:\n\n" + formatted_chunks + "\n\n"
        "Why might this investor vote AGAINST this resolution? Please include specific references to the company report and the investor policy."
    )
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert in corporate governance and ESG voting."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(GPT error: {html.escape(str(e))})"


# -----------------------------
# Routes
# -----------------------------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/login", response_class=HTMLResponse)
def login_page():
    return """
    <html><body>
    <h2>Login</h2>
    <form method='post' action='/login'>
        <label>Username: <input type='text' name='username'></label><br><br>
        <label>Password: <input type='password' name='password'></label><br><br>
        <input type='submit' value='Login'>
    </form>
    </body></html>
    """


@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    if secrets.compare_digest(username, USERNAME) and secrets.compare_digest(password, PASSWORD):
        token = secrets.token_urlsafe(16)
        sessions[token] = username
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(
            key="session",
            value=token,
            httponly=True,
            secure=True,
            samesite="lax",
        )
        return response
    return HTMLResponse("<h3>Invalid credentials. <a href='/login'>Try again</a></h3>", status_code=401)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    user = login_required(request)
    if isinstance(user, Response):
        return user

    options_html = "".join(
        f"<option value=\"{html.escape(inv)}\">{html.escape(inv)}</option>"
        for inv in investor_policies.keys()
    )

    return """
    <html><body>
    <h2>Investor Vote Explanation Tool</h2>
    <form id='analyzeForm' action='/upload' method='post' enctype='multipart/form-data'>
        <input type='file' name='file' accept='.docx,.pdf'><br><br>
        <label>Select Investor:</label><br>
        <select name='policy'>
            <option value='all'>All</option>
            %s
        </select><br><br>
        <input type='submit' value='Analyse'>
    </form>

    <div id='loader' style='display:none;'>⏳ Analysing...</div>
    <div id='results'></div>

    <br>
    <button type="button" onclick="exportCSV()">📄 Export CSV</button>

    <script>
      function toCSVCell(v) {
        if (v == null) return '';
        const s = String(v).replaceAll('"', '""');
        return '"' + s + '"';
      }

      function exportCSV() {
        try {
          const blocks = document.querySelectorAll('.result-block');
          if (!blocks || blocks.length === 0) {
            alert('No results to export yet. Run an analysis first.');
            return;
          }
          const rows = [['Investor','Verdict']];
          blocks.forEach(block => {
            const investor = block.getAttribute('data-investor') || '';
            const verdict  = block.getAttribute('data-verdict')  || '';
            rows.push([investor, verdict]);
          });

          const csv = '\\ufeff' + rows.map(r => r.map(toCSVCell).join(',')).join('\\n');
          const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
          const url  = URL.createObjectURL(blob);
          const a    = document.createElement('a');
          a.href = url;
          a.download = 'analysis_results.csv';
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          setTimeout(() => URL.revokeObjectURL(url), 500);
        } catch (e) {
          console.error('Export failed:', e);
          alert('Export failed. Check the console for details.');
        }
      }
    </script>

    <script>
    document.getElementById('analyzeForm').onsubmit = async function(e) {
        e.preventDefault();
        const form = e.target;
        const formData = new FormData(form);
        document.getElementById('loader').style.display = 'block';
        document.getElementById('results').innerHTML = '';
        const response = await fetch('/upload', { method: 'POST', body: formData, credentials: 'include' });
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value, { stream: true });
            document.getElementById('results').innerHTML += chunk;
        }
        document.getElementById('loader').style.display = 'none';
    };
    </script>

    </body></html>
    """ % options_html


@app.post("/upload", response_class=StreamingResponse)
def upload_file(request: Request, file: UploadFile = File(...), policy: str = Form(...)):
    user = login_required(request)
    if isinstance(user, Response):
        return user

    contents = file.file.read()
    filename = (file.filename or "").lower()

    # Infer company from filename and load CSV of "against" investors
    base = os.path.splitext(os.path.basename(filename))[0]
    company_key = None
    if "autotrader" in base:
        company_key = "autotrader"
    elif "unilever" in base:
        company_key = "unilever"
    elif "leg" in base:
        company_key = "leg"
    elif "asr" in base:
        company_key = "asr"
    elif "euronext" in base:
        company_key = "euronext"
    elif "sainsbury" in base or "sainsbury's" in base or "j sainsbury" in base:
        company_key = "sainsbury"

    csv_force_reason_investors: set[str] = set()
    if company_key:
        csv_path = CSV_MAP.get(company_key)
        if csv_path and os.path.exists(csv_path):
            try:
                csv_force_reason_investors = load_company_against_investors_from_csv(csv_path)
                print(f"[CSV] Matched {len(csv_force_reason_investors)} investors from {csv_path}")
            except Exception as _e:
                print(f"[CSV] Failed to load {csv_path}: {_e}")
        else:
            print(f"[CSV] No CSV available or path missing for company '{company_key}'")

    def stream():
        try:
            if filename.endswith(".docx"):
                full_text = extract_text_from_docx_bytes(contents)
            elif filename.endswith(".pdf"):
                full_text = extract_text_from_pdf_bytes(contents)
            else:
                yield f"<p>Unsupported file type: {html.escape(filename)}. Please upload .docx or .pdf.</p>"
                return
        except Exception as e:
            yield f"<p>Error extracting text: {html.escape(str(e))}</p>"
            return

        if not full_text.strip():
            yield "<p>No readable text found in document.</p>"
            return

        yield "<p>✅ Text extracted.</p>"
        yield "<p>✂️ Chunking…</p>"

        yield f"<p>⚙️ Computing embeddings & classifying for {html.escape(policy)}…</p>"

        chunks = chunk_text(full_text)
        if not chunks:
            yield "<p>Document is too short to chunk.</p>"
            return

        chunk_embeddings = get_embeddings(chunks, batch_size=64)
        yield f"<p>🔹 Running selection & classification for {html.escape(policy)}…</p>"

        def analyse_investor(name, investor_policy, force_reason=False):
            policy_emb = get_embedding(investor_policy)
            sims = chunk_embeddings @ policy_emb
            top_idx = np.argsort(sims)[-TOP_K:][::-1]
            top_chunks = [chunks[i] for i in top_idx]
            top_sims = sims[top_idx]

            scored = [(c, *predict_vote(investor_policy, c)) for c in top_chunks]
            maj, conf, frac, mean_prob = weighted_decision(scored, top_sims)

            maj_display = AGAINST_LABEL if bool(force_reason) else maj
            verdict = "AGAINST" if maj_display == AGAINST_LABEL else "FOR"

            need_reason = maj_display == AGAINST_LABEL
            reason_html = ""
            if need_reason:
                if client is None or not OPENAI_API_KEY:
                    reason_text = "OpenAI key not set — set OPENAI_API_KEY to see reasons"
                else:
                    top_chunk_texts = [c for c, _, _ in scored]
                    gpt_text = get_gpt_reason(investor_policy, top_chunk_texts)
                    reason_text = gpt_text or "(No explanation returned)"
                reason_html = (
                    "<div style='background:#f7f7f7;padding:10px;border-left:4px solid #cc0000;"
                    "margin-top:6px;color:#333;'><b>Reason:</b><br>" + escape_html(reason_text) + "</div>"
                )

            yield (
                f"<div class='result-block' data-investor='{html.escape(name, quote=True)}' "
                f"data-verdict='{html.escape(verdict, quote=True)}'>"
                f"<h3>Investor: {html.escape(name)}</h3>"
                f"<h4>{'❌ AGAINST' if maj_display == AGAINST_LABEL else '✅ FOR'}</h4>"
                f"{reason_html}"
                f"<hr></div>"
            )

        if policy.lower() == "all":
            for inv, pol in investor_policies.items():
                yield from analyse_investor(inv, pol, force_reason=(inv in csv_force_reason_investors))
        else:
            pol = investor_policies.get(policy)
            if not pol:
                yield f"<p>Unknown investor {html.escape(policy)}</p>"
                return
            yield from analyse_investor(policy, pol, force_reason=(policy in csv_force_reason_investors))

    return StreamingResponse(stream(), media_type="text/html")
