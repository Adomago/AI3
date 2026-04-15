"""
Philippine Labor Code & Employee Rights Assistant
Multi-Model RAG Evaluation: Qwen2.5-7B | LLaMA-3.1-8B | Gemma-2-9B

Gradio deployment with ZeroGPU support on Hugging Face Spaces.
"""

import os
import re
import gc
import time
import warnings
import traceback
import nltk
import torch
import faiss
import spaces
import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from huggingface_hub import login, snapshot_download
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# HF Authentication
# ---------------------------------------------------------------------------
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    print("HF token login successful.")
else:
    print("WARNING: HF_TOKEN not set. Gated models (LLaMA, Gemma) will fail.")

# ---------------------------------------------------------------------------
# NLTK data
# ---------------------------------------------------------------------------
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PDF_PATH             = "laborcode.pdf"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME  = "cross-encoder/ms-marco-MiniLM-L12-v2"

CHUNK_MAX_LEN = 1200
CHUNK_OVERLAP = 200
CHUNK_MIN_LEN = 100

MODEL_CONFIGS = {
    "Qwen2.5-7B-Instruct": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "supports_system": True,
    },
    "LLaMA-3.1-8B-Instruct": {
        "hf_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "supports_system": True,
    },
    "Gemma-2-9B-IT": {
        "hf_id": "google/gemma-2-9b-it",
        "supports_system": False,
    },
}

# ---------------------------------------------------------------------------
# FIX 1 — Language detection helper + iron-clad language instruction
#
# The core bug: models were defaulting to Tagalog because the corpus is
# Philippine-themed and the language rule was buried. Fix:
#   a) Detect the question language in Python (fast, no GPU cost).
#   b) Pass an EXPLICIT, unambiguous instruction as the very FIRST line
#      of both the system prompt and the user turn so it cannot be ignored.
# ---------------------------------------------------------------------------

# Simple heuristic: if the question contains common Tagalog function words
# that do NOT appear in normal English, we treat it as Filipino.
_TAGALOG_MARKERS = re.compile(
    r"\b(ano|alin|sino|paano|bakit|kailan|saan|magkano|ilan|"
    r"ng|mga|sa|na|at|ay|ang|ito|iyon|siya|sila|kayo|kami|tayo|"
    r"po|ho|ba|raw|daw|lang|lamang|naman|yung|yun|kasi|talaga|"
    r"meron|mayroon|wala|hindi|oo|hindi|pati|pero|dahil|kung|"
    r"trabaho|sahod|batas|karapatan|tanggalin|artikulo|sweldo)\b",
    re.IGNORECASE,
)

def detect_language(text: str) -> str:
    """Return 'filipino' or 'english'."""
    tokens = text.lower().split()
    hits = sum(1 for t in tokens if _TAGALOG_MARKERS.match(t))
    # If >25 % of tokens are Tagalog markers, treat as Filipino
    return "filipino" if tokens and (hits / len(tokens)) > 0.25 else "english"


def _build_system_prompt(lang: str) -> str:
    """
    Put the language rule on line 1, before everything else.
    This is the most reliable way to prevent models from ignoring it.
    """
    if lang == "filipino":
        lang_rule = (
            "⚠️ LANGUAGE RULE (HIGHEST PRIORITY): The user asked in Filipino/Tagalog. "
            "You MUST respond ENTIRELY in Filipino/Tagalog. "
            "Do NOT use English anywhere in your response. "
            "Every word — citations, explanations, headings — must be in Filipino/Tagalog.\n\n"
        )
    else:
        lang_rule = (
            "⚠️ LANGUAGE RULE (HIGHEST PRIORITY): The user asked in English. "
            "You MUST respond ENTIRELY in English. "
            "Do NOT use Filipino/Tagalog anywhere in your response. "
            "Every word — citations, explanations, headings — must be in English.\n\n"
        )

    body = (
        "You are Lex, a highly knowledgeable Philippine Labor Law assistant. "
        "Your role is to provide thorough, well-structured answers about the Philippine Labor Code "
        "(Presidential Decree No. 442, as amended).\n\n"
        "When answering:\n"
        "1. Always cite the exact Article number(s) (e.g., 'Under Article 86...').\n"
        "2. Quote or paraphrase the relevant provision directly.\n"
        "3. Explain what it means in plain language.\n"
        "4. If multiple articles apply, address each one.\n"
        "5. If the context does not contain enough information, say so honestly — do not fabricate.\n"
        "6. Be concise but complete. Avoid unnecessary filler."
    )
    return lang_rule + body


JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator of AI-generated legal answers about the Philippine Labor Code.\n"
    "Score the ANSWER on a scale from 0.0 to 1.0 using this rubric:\n"
    "  1.0 — Complete, accurate, cites the correct Article(s), directly answers the question\n"
    "  0.8 — Mostly accurate with a minor omission or imprecise citation\n"
    "  0.6 — Partially correct, answers part of the question\n"
    "  0.4 — Vague or incomplete, lacking citations\n"
    "  0.2 — Largely incorrect or hallucinated content\n"
    "  0.0 — Completely wrong, refused to answer, or irrelevant\n\n"
    "Respond with ONLY a single decimal number between 0.0 and 1.0. Nothing else."
)

EXAMPLE_QUESTIONS = [
    "What are the just causes for termination by the employer?",
    "What is the rule on overtime pay and when is it required?",
    "How many days of service incentive leave is an employee entitled to?",
    "Ano ang night shift differential at magkano ito?",
    "What are the authorized causes for retrenchment or lay-off?",
]

# ---------------------------------------------------------------------------
# Pre-download models to cache during CPU startup
# ---------------------------------------------------------------------------
CACHED_MODELS: set = set()

print("Pre-downloading models to cache (CPU startup — internet available)...")
for _name, _cfg in MODEL_CONFIGS.items():
    _ok = False
    for _attempt in range(1, 4):
        try:
            snapshot_download(
                repo_id=_cfg["hf_id"],
                token=hf_token,
                ignore_patterns=["*.bin", "*.pt", "original/*"],
            )
            CACHED_MODELS.add(_name)
            print(f"  ✓ {_name} cached (attempt {_attempt})")
            _ok = True
            break
        except Exception as _e:
            print(f"  ! {_name} attempt {_attempt}/3 failed: {_e}")
            if _attempt < 3:
                time.sleep(5)
    if not _ok:
        print(f"  ✗ {_name} could not be cached — will be skipped at inference.")

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
print(f"Offline mode enabled. Cached models: {CACHED_MODELS}")

# ---------------------------------------------------------------------------
# Greeting detection
# ---------------------------------------------------------------------------
GREETING_PATTERNS = [
    r"^(hi|hello|hey|good morning|good afternoon|good evening|kamusta|kumusta|musta|helo|oi|uy)[\s!?.]*$",
    r"^(what can you do|what are you|who are you|what is lex|are you a bot|are you ai)[\s?]*$",
    r"^(thanks?|thank you|salamat|maraming salamat|ty)[\s!.]*$",
    r"^(bye|goodbye|see you|paalam|ok|okay|sure|alright|got it|noted)[\s!.]*$",
    r"^(help|tulong|tulungan mo ako)[\s!?]*$",
]

GREETING_RESPONSE = (
    "Hello! I am Lex, your Philippine Labor Law assistant. "
    "Feel free to ask me any questions about labor rights, employment policies, "
    "wages, working hours, leaves, termination, or any workplace concerns under "
    "the Philippine Labor Code (PD 442). How can I help you today?"
)

def is_greeting(text: str) -> bool:
    t = text.strip().lower()
    for pat in GREETING_PATTERNS:
        if re.match(pat, t, re.IGNORECASE):
            return True
    legal_keywords = [
        "article", "labor", "wage", "leave", "work", "employ",
        "salary", "pay", "overtime", "holiday", "terminate",
        "strike", "union", "dole", "law", "code", "right",
        "benefit", "retire", "resign", "dismiss",
        "artikulo", "trabaho", "sahod", "batas", "karapatan",
        "tanggalin", "buwis", "regular", "kontrata",
    ]
    tokens = t.split()
    if len(tokens) <= 3 and not any(kw in t for kw in legal_keywords):
        return True
    return False

# ---------------------------------------------------------------------------
# PDF processing and chunking
# ---------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def clean_text(text: str) -> str:
    text = re.sub(r"(ART\.)\s*(\d+)\s+(\d+)\.", r"\1 \2\3.", text)
    text = re.sub(r"(Article\s+)(\d+)\s+(\d+)", r"\1\2\3", text)
    text = re.sub(r"---\s*Page\s*\d+\s*---", "", text)
    text = re.sub(
        r"\n\s*\d{1,3}\s+(?:See|As amended|R\.A\.|P\.D\.|E\.O\.|The |This |Pursuant|Section|Sec\.).*",
        "", text, flags=re.IGNORECASE,
    )
    text = re.sub(r"\[Footnote\].*?\n", "\n", text, flags=re.DOTALL)
    text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", text)
    text = re.sub(r"[ \t]{3,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def is_substantive_chunk(chunk: str, max_footnote_ratio: float = 0.08) -> bool:
    footnote_markers = [
        "[Footnote]", "See DOLE", "As amended by", "superseded by",
        "cross-reference", "R.A. No.", "P.D. No.", "E.O. No.",
        "pursuant to", "inserted in", "renumbered as",
    ]
    words = chunk.split()
    if not words:
        return False
    hits = sum(chunk.lower().count(m.lower()) for m in footnote_markers)
    return (hits / len(words)) < max_footnote_ratio

def fix_broken_article_header(chunk: str) -> str:
    return re.sub(
        r"(ART\.?\s*)(\d)\s+(\d+\.)",
        lambda m: m.group(1) + m.group(2) + m.group(3),
        chunk, flags=re.IGNORECASE,
    )

def chunk_text_by_article(text: str,
                           max_len: int = CHUNK_MAX_LEN,
                           overlap: int = CHUNK_OVERLAP,
                           min_len: int = CHUNK_MIN_LEN):
    article_pattern = re.compile(
        r"(?=(?:ART\.|Art\.|ARTICLE)\s+\d+[\.\ ])", re.IGNORECASE
    )
    raw_splits = article_pattern.split(text)
    chunks = []
    for block in raw_splits:
        block = block.strip()
        if not block:
            continue
        if len(block) <= max_len:
            if len(block) >= min_len:
                chunks.append(block)
        else:
            header_match = re.match(r"((?:ART\.|Art\.|ARTICLE)\s+\d+[^.]*\.)", block)
            header   = header_match.group(1).strip() if header_match else ""
            sentences = re.split(r"(?<=[.!?;])\s+", block)
            current, chunk_num = "", 0
            for sent in sentences:
                if len(current) + len(sent) > max_len:
                    if current:
                        chunks.append(current.strip())
                        chunk_num += 1
                        tail    = current[-overlap:] if len(current) > overlap else ""
                        current = (header + " [cont] " + tail) if header and chunk_num > 0 else tail
                current += " " + sent
            if current.strip() and len(current.strip()) >= min_len:
                chunks.append(current.strip())

    boilerplate = [
        "NOT FOR SALE", "Copyright", "SILVESTRE H. BELLO",
        "Table of Contents", "FOREWORD", "www.dole.gov.ph",
        "Repealing Clause", "cross-references all superseded", "Name of Decree",
    ]
    chunks = [
        c for c in chunks
        if not any(b.lower() in c.lower() for b in boilerplate)
        and len(c.strip()) > min_len
        and is_substantive_chunk(c)
    ]
    return [fix_broken_article_header(c) for c in chunks]

# ---------------------------------------------------------------------------
# Retrieval
# FIX 2 — Reduced final_k to 4 (saves reranker + embedding time each query)
# ---------------------------------------------------------------------------
def mmr_select(candidates, scores, embeddings, k=4, lam=0.6):
    if len(candidates) <= k:
        return candidates, scores
    embs = np.array(embeddings)
    selected_idx, remaining = [], list(range(len(candidates)))
    while len(selected_idx) < k and remaining:
        if not selected_idx:
            best = max(remaining, key=lambda i: scores[i])
        else:
            sel_embs = embs[selected_idx]
            def mmr_score(i, _s=sel_embs):
                return lam * scores[i] - (1.0 - lam) * float(np.max(_s @ embs[i]))
            best = max(remaining, key=mmr_score)
        selected_idx.append(best)
        remaining.remove(best)
    return [candidates[i] for i in selected_idx], [scores[i] for i in selected_idx]

def deduplicate_by_article(ranked_pairs, max_per_article=2, final_k=4):
    seen_art, final = {}, []
    for chunk, score in ranked_pairs:
        match = re.match(r"(ART\.?\s*\d+)", chunk, re.IGNORECASE)
        key   = match.group(1).upper().replace(" ", "") if match else "UNK"
        count = seen_art.get(key, 0)
        if count < max_per_article:
            final.append((chunk, score))
            seen_art[key] = count + 1
        if len(final) == final_k:
            break
    return final

def hybrid_retrieve_and_rerank(question, embedder, faiss_index, bm25_index,
                                reranker, doc_chunks,
                                initial_k=15, rerank_k=6, final_k=4):
    # FIX 2: initial_k 20→15, rerank_k 8→6, final_k 5→4
    query_emb = embedder.encode(
        [f"query: {question}"], convert_to_numpy=True, normalize_embeddings=True
    )
    _, dense_indices = faiss_index.search(query_emb, initial_k)
    dense_ranking    = list(dense_indices[0])

    bm25_raw     = bm25_index.get_scores(question.lower().split())
    bm25_ranking = list(np.argsort(bm25_raw)[::-1][:initial_k])

    rrf_k, rrf_scores = 60, {}
    for rank, idx in enumerate(dense_ranking):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rank + rrf_k)
    for rank, idx in enumerate(bm25_ranking):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rank + rrf_k)

    fused_indices    = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:initial_k]
    candidate_chunks = [doc_chunks[i] for i in fused_indices]

    rerank_scores_arr = reranker.predict([[question, c] for c in candidate_chunks])
    ranked_all = sorted(
        zip(candidate_chunks, rerank_scores_arr.tolist()),
        key=lambda x: x[1], reverse=True,
    )[:rerank_k]

    deduped      = deduplicate_by_article(ranked_all, max_per_article=2, final_k=rerank_k)
    dedup_chunks = [x[0] for x in deduped]
    dedup_scores = [x[1] for x in deduped]

    cand_embs = embedder.encode(
        [f"passage: {c}" for c in dedup_chunks],
        convert_to_numpy=True, normalize_embeddings=True,
    )
    return mmr_select(dedup_chunks, dedup_scores, cand_embs, k=final_k, lam=0.6)

# ---------------------------------------------------------------------------
# Citation / evaluation helpers
# ---------------------------------------------------------------------------
def _extract_articles(text: str) -> set:
    return set(re.findall(
        r"(?:ART\.|Art\.|Article|ARTICLE)\s*(\d+)", text, re.IGNORECASE
    ))

def _get_expected_articles(top_chunks: list, top_scores: list) -> set:
    scored = sorted(zip(top_scores, top_chunks), reverse=True)
    relevant = [(s, c) for s, c in scored if s > 0.0][:2]
    if not relevant:
        relevant = [scored[0]] if scored else []
    expected = set()
    for _, chunk in relevant:
        expected |= _extract_articles(chunk)
    return expected

def compute_citation_accuracy(answer: str, top_chunks: list, top_scores: list) -> float:
    expected = _get_expected_articles(top_chunks, top_scores)
    if not expected:
        return 1.0
    found = sum(
        1 for art in expected
        if any(re.search(p, answer, re.IGNORECASE) for p in [
            rf"Article\s*{art}\b", rf"Art\.?\s*{art}\b",
            rf"ART\.?\s*{art}\b",  rf"Artikulo\s*{art}\b",
        ])
    )
    return found / len(expected)

def compute_faithfulness(answer: str, context_chunks: list, embedder) -> float:
    if not answer or not context_chunks:
        return 0.0
    combined = " ".join(context_chunks)
    embs = embedder.encode([answer, combined], convert_to_numpy=True, normalize_embeddings=True)
    return float(cos_sim([embs[0]], [embs[1]])[0][0])

def compute_semantic_sim(answer: str, question: str, embedder) -> float:
    if not answer or not question:
        return 0.0
    embs = embedder.encode([answer, question], convert_to_numpy=True, normalize_embeddings=True)
    return float(cos_sim([embs[0]], [embs[1]])[0][0])

def compute_answer_relevancy(answer: str, question: str, embedder) -> float:
    if not answer or not question:
        return 0.0
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 10]
    if not sentences:
        return compute_semantic_sim(answer, question, embedder)
    q_emb  = embedder.encode([question],  convert_to_numpy=True, normalize_embeddings=True)
    s_embs = embedder.encode(sentences,   convert_to_numpy=True, normalize_embeddings=True)
    return float(np.mean([float(cos_sim([q_emb[0]], [s])[0][0]) for s in s_embs]))

def compute_recall_at_k(top_chunks: list, top_scores: list) -> float:
    expected = _get_expected_articles(top_chunks, top_scores)
    if not expected:
        return 1.0
    all_text = " ".join(top_chunks)
    found = sum(
        1 for art in expected
        if any(re.search(p, all_text, re.IGNORECASE) for p in [
            rf"Article\s*{art}\b", rf"Art\.?\s*{art}\b", rf"ART\.?\s*{art}\b",
        ])
    )
    return found / len(expected)

def compute_precision_at_k(top_scores: list) -> float:
    if not top_scores:
        return 0.0
    return sum(1 for s in top_scores if s > 0.0) / len(top_scores)

def evaluate_answer(question: str, answer: str,
                    top_chunks: list, top_scores: list, embedder) -> dict:
    return {
        "Faithfulness":      round(compute_faithfulness(answer, top_chunks, embedder), 4),
        "Semantic Sim":      round(compute_semantic_sim(answer, question, embedder), 4),
        "Answer Relevancy":  round(compute_answer_relevancy(answer, question, embedder), 4),
        "Citation Accuracy": round(compute_citation_accuracy(answer, top_chunks, top_scores), 4),
        "Recall@4":          round(compute_recall_at_k(top_chunks, top_scores), 4),
        "Precision@4":       round(compute_precision_at_k(top_scores), 4),
    }

# ---------------------------------------------------------------------------
# LLM-as-a-Judge
# FIX 3 — max_new_tokens cut to 8 (just needs one number), saves ~2–3s/model
# ---------------------------------------------------------------------------
def run_llm_judge(question: str, answer: str, context_chunks: list,
                  model, tokenizer, supports_system: bool) -> float:
    try:
        # Trim context to 800 chars (was 1500) — judge doesn't need full context
        context_snippet = "\n\n".join(context_chunks[:2])[:800]
        q_and_a = (
            f"QUESTION: {question}\n\n"
            f"CONTEXT:\n{context_snippet}\n\n"
            f"ANSWER:\n{answer[:600]}\n\n"   # cap answer too
            f"Score:"
        )
        if supports_system:
            messages = [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user",   "content": q_and_a},
            ]
        else:
            messages = [{"role": "user",
                         "content": f"{JUDGE_SYSTEM_PROMPT}\n\n{q_and_a}"}]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=8,   # FIX 3: was 12, only need "0.8" etc.
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        raw  = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()
        nums = re.findall(r"\d+\.?\d*", raw)
        if nums:
            val = float(nums[0])
            if val > 1.0:
                val = val / 10.0 if val <= 10.0 else 1.0
            return min(max(round(val, 2), 0.0), 1.0)
    except Exception:
        traceback.print_exc()
    return 0.0

# ---------------------------------------------------------------------------
# Chart & table rendering  (renamed @5 → @4)
# ---------------------------------------------------------------------------
CHART_METRICS = [
    "Faithfulness", "Semantic Sim", "Answer Relevancy",
    "Citation Accuracy", "Recall@4", "Precision@4", "LLM-Judge",
]
TABLE_METRICS = CHART_METRICS + ["Latency (s)"]

def render_comparison_chart(all_metrics: dict) -> plt.Figure:
    model_names = list(all_metrics.keys())
    n_metrics   = len(CHART_METRICS)
    n_models    = len(model_names)
    x           = np.arange(n_metrics)
    width       = 0.8 / max(n_models, 1)
    colors      = ["#2563eb", "#dc2626", "#16a34a"]

    fig, ax = plt.subplots(figsize=(15, 6))
    for i, model in enumerate(model_names):
        values = [all_metrics[model].get(m, 0.0) for m in CHART_METRICS]
        offset = (i - n_models / 2 + 0.5) * width
        bars   = ax.bar(x + offset, values, width, label=model, color=colors[i % 3])
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7,
            )
    ax.set_ylabel("Score")
    ax.set_title("Multi-Model RAG Evaluation — Philippine Labor Code (PD 442)")
    ax.set_xticks(x)
    ax.set_xticklabels(CHART_METRICS, rotation=25, ha="right")
    ax.set_ylim(0, 1.18)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig

def build_metrics_table(all_metrics: dict) -> str:
    if not all_metrics:
        return ""
    model_names = list(all_metrics.keys())
    header = "| Metric | " + " | ".join(model_names) + " |"
    sep    = "|---|" + "|".join(["---"] * len(model_names)) + "|"
    rows   = []
    for key in TABLE_METRICS:
        row = f"| **{key}** |"
        for mn in model_names:
            val = all_metrics[mn].get(key, "—")
            row += f" {val:.4f} |" if isinstance(val, float) else f" {val} |"
        rows.append(row)
    return "\n".join([header, sep] + rows)

# ---------------------------------------------------------------------------
# Load retrieval infrastructure at startup (CPU only)
# ---------------------------------------------------------------------------
print("Loading PDF and building chunks...")
if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"'{PDF_PATH}' not found.")

raw_text     = extract_text_from_pdf(PDF_PATH)
cleaned_text = clean_text(raw_text)
CHUNKS       = chunk_text_by_article(cleaned_text)
print(f"Created {len(CHUNKS)} chunks  "
      f"(max_len={CHUNK_MAX_LEN}, overlap={CHUNK_OVERLAP}, min_len={CHUNK_MIN_LEN})")

print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
EMBEDDER = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")

print("Building FAISS index...")
prefixed_chunks  = [f"passage: {c}" for c in CHUNKS]
chunk_embeddings = EMBEDDER.encode(
    prefixed_chunks, batch_size=16, show_progress_bar=True,
    convert_to_numpy=True, normalize_embeddings=True,
)
FAISS_INDEX = faiss.IndexFlatIP(chunk_embeddings.shape[1])
FAISS_INDEX.add(chunk_embeddings)

print("Building BM25 index...")
BM25_INDEX = BM25Okapi([c.lower().split() for c in CHUNKS])

print("Loading cross-encoder reranker...")
RERANKER = CrossEncoder(RERANKER_MODEL_NAME, device="cpu")
print("Retrieval infrastructure ready.")

# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------
def _cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# ---------------------------------------------------------------------------
# Single-model generation
#
# FIX 1 (language): detect language before building prompt, inject explicit
#   language rule as the very first line of the system/user message.
# FIX 4 (speed): max_new_tokens 512→320, max prompt length 4096→3072,
#   context capped at 3 chunks (was all 5), context chars capped at 2400.
# ---------------------------------------------------------------------------
def _generate_single(model_name: str, question: str,
                     context_chunks: list, context_scores: list,
                     lang: str) -> dict:
    if model_name not in CACHED_MODELS:
        return {
            "answer": (
                f"[{model_name} unavailable: model files were not downloaded at startup. "
                "Check that HF_TOKEN is set and you have accepted the model license.]"
            ),
            "latency": 0.0, "latency_detail": "not cached", "llm_judge": 0.0,
        }

    config = MODEL_CONFIGS[model_name]
    try:
        load_start = time.time()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config["hf_id"], trust_remote_code=True,
            token=hf_token, local_files_only=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            config["hf_id"], quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True,
            torch_dtype=torch.float16, token=hf_token, local_files_only=True,
        )
        model.eval()
        load_time = time.time() - load_start

        # ---- Build prompt with language-aware system prompt ----
        system_prompt = _build_system_prompt(lang)

        # FIX 4: use only top-3 chunks and cap total context at 2400 chars
        top3_chunks   = context_chunks[:3]
        context_block = "\n\n---\n\n".join(top3_chunks)[:2400]

        # FIX 1: repeat the language rule at the END of the user turn too —
        # this "sandwiches" the instruction so models cannot ignore it.
        if lang == "filipino":
            lang_reminder = "\n\nMULTING PAALALA: Sumagot nang BUO sa Filipino/Tagalog lamang."
        else:
            lang_reminder = "\n\nFINAL REMINDER: Respond ENTIRELY in English only."

        user_content = (
            f"CONTEXT (from the Philippine Labor Code):\n{context_block}\n\n"
            f"QUESTION: {question}\n\n"
            "Write a thorough answer. Cite every applicable Article number explicitly. "
            "Quote the key provision and explain it in plain language."
            f"{lang_reminder}"
        )

        if config["supports_system"]:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ]
        else:
            # Gemma: merge system into user turn
            messages = [{"role": "user",
                         "content": f"{system_prompt}\n\n{user_content}"}]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # FIX 4: reduced max input length 4096→3072 to leave more room to skip
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        inf_start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=320,     # FIX 4: was 512 — saves ~30–40% inference time
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.pad_token_id,
            )
        inf_time = time.time() - inf_start

        input_len = inputs["input_ids"].shape[1]
        answer    = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

        # LLM-as-a-Judge
        llm_judge = run_llm_judge(
            question, answer, context_chunks,
            model, tokenizer, config["supports_system"],
        )

        del inputs, outputs
        _cleanup_gpu()
        total_time     = load_time + inf_time
        latency_detail = f"load={load_time:.1f}s  infer={inf_time:.1f}s  total={total_time:.1f}s"
        del model, tokenizer
        _cleanup_gpu()

        return {
            "answer": answer, "latency": round(total_time, 2),
            "latency_detail": latency_detail, "llm_judge": llm_judge,
        }

    except Exception as e:
        traceback.print_exc()
        _cleanup_gpu()
        return {"answer": f"[Error: {str(e)}]", "latency": 0.0,
                "latency_detail": "error", "llm_judge": 0.0}


@spaces.GPU(duration=80)
def generate_qwen(question, context_chunks, context_scores, lang):
    return _generate_single("Qwen2.5-7B-Instruct", question, context_chunks, context_scores, lang)

@spaces.GPU(duration=80)
def generate_llama(question, context_chunks, context_scores, lang):
    return _generate_single("LLaMA-3.1-8B-Instruct", question, context_chunks, context_scores, lang)

@spaces.GPU(duration=80)
def generate_gemma(question, context_chunks, context_scores, lang):
    return _generate_single("Gemma-2-9B-IT", question, context_chunks, context_scores, lang)

# ---------------------------------------------------------------------------
# Streaming query handler
# ---------------------------------------------------------------------------
def process_query(question: str):
    if not question or not question.strip():
        yield ("Please enter a question.", "", "", "", "", "", None)
        return

    question = question.strip()

    if is_greeting(question):
        yield (
            GREETING_RESPONSE,
            GREETING_RESPONSE,
            GREETING_RESPONSE,
            "",
            "Greeting detected — no retrieval needed.",
            "",
            None,
        )
        return

    # Detect language ONCE here, pass to every model
    lang = detect_language(question)
    print(f"[Language detected: {lang}] Question: {question[:80]}")

    # ---- Retrieval (CPU) ----
    t0 = time.time()
    top_chunks, top_scores = hybrid_retrieve_and_rerank(
        question=question, embedder=EMBEDDER, faiss_index=FAISS_INDEX,
        bm25_index=BM25_INDEX, reranker=RERANKER, doc_chunks=CHUNKS,
        initial_k=15, rerank_k=6, final_k=4,
    )
    retrieval_time = time.time() - t0

    expected_arts = _get_expected_articles(top_chunks, top_scores)
    arts_str      = ", ".join(sorted(expected_arts, key=lambda x: int(x))) if expected_arts else "none detected"

    chunks_display = ""
    for i, (chunk, score) in enumerate(zip(top_chunks, top_scores)):
        chunks_display += f"--- Chunk {i+1} | Reranker Score: {score:.4f} ---\n{chunk}\n\n"
    chunks_display += (
        f"[Retrieval: {retrieval_time:.2f}s | Language: {lang}]\n"
        f"[Citation Accuracy expected articles: {arts_str}]"
    )

    yield ("⏳ Generating...", "⏳ Generating...", "⏳ Generating...",
           chunks_display, "Generating responses...", "", None)

    # ---- Generation ----
    all_results: dict = {}
    all_metrics: dict = {}
    display: dict     = {
        "qwen":  "⏳ Generating...",
        "llama": "⏳ Generating...",
        "gemma": "⏳ Generating...",
    }

    gen_sequence = [
        ("Qwen2.5-7B-Instruct",   generate_qwen,  "qwen"),
        ("LLaMA-3.1-8B-Instruct", generate_llama, "llama"),
        ("Gemma-2-9B-IT",         generate_gemma, "gemma"),
    ]

    for model_name, gen_fn, key in gen_sequence:
        result = gen_fn(question, top_chunks, top_scores, lang)
        all_results[model_name] = result
        display[key] = result["answer"]

        answer = result["answer"]
        if answer and not answer.startswith("["):
            metrics = evaluate_answer(question, answer, top_chunks, top_scores, EMBEDDER)
            metrics["LLM-Judge"]   = round(result.get("llm_judge", 0.0), 4)
            metrics["Latency (s)"] = round(result.get("latency",   0.0), 2)
            all_metrics[model_name] = metrics

        lat_parts = []
        for mn, gkey in [("Qwen2.5-7B-Instruct",   "qwen"),
                         ("LLaMA-3.1-8B-Instruct",  "llama"),
                         ("Gemma-2-9B-IT",           "gemma")]:
            if mn in all_results:
                detail = all_results[mn].get("latency_detail", "")
                lat_parts.append(f"{mn}: {detail} ✓")
            else:
                lat_parts.append(f"{mn}: pending...")
        latency_info = "\n".join(lat_parts)

        metrics_md = build_metrics_table(all_metrics) if all_metrics else ""
        chart_fig  = render_comparison_chart(
            {mn: {k: v for k, v in m.items() if k != "Latency (s)"}
             for mn, m in all_metrics.items()}
        ) if all_metrics else None

        yield (
            display["qwen"],
            display["llama"],
            display["gemma"],
            chunks_display,
            latency_info,
            metrics_md,
            chart_fig,
        )

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
CSS = ".example-btn { font-size:0.82rem !important; }"

with gr.Blocks(title="PH Labor Code RAG Assistant", theme=gr.themes.Base(), css=CSS) as demo:

    gr.Markdown(
        "# 🇵🇭 Philippine Labor Code & Employee Rights Assistant\n"
        "### Multi-Model RAG Evaluation: Qwen2.5-7B · LLaMA-3.1-8B · Gemma-2-9B\n\n"
        "Ask any question about the Philippine Labor Code (PD 442) — in English or Filipino. "
        "Each model's answer and scores appear **as soon as that model finishes**."
    )
    gr.Markdown("---")

    gr.Markdown("#### 💡 Try an example:")
    with gr.Row():
        example_btns = [gr.Button(q, elem_classes=["example-btn"]) for q in EXAMPLE_QUESTIONS]

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=4):
            question_input = gr.Textbox(
                label="Your Question (English or Filipino)",
                placeholder="e.g. What are the just causes for termination? / Ano ang night shift differential?",
                lines=2,
            )
        with gr.Column(scale=1):
            submit_btn = gr.Button("Submit", variant="primary", size="lg")

    gr.Markdown("---")
    gr.Markdown("## 🤖 Model Responses")

    with gr.Row():
        qwen_output  = gr.Textbox(label="Qwen2.5-7B-Instruct",   lines=14)
        llama_output = gr.Textbox(label="LLaMA-3.1-8B-Instruct", lines=14)
        gemma_output = gr.Textbox(label="Gemma-2-9B-IT",          lines=14)

    latency_display = gr.Textbox(
        label="Generation Latency (load / inference / total per model)",
        interactive=False, lines=3,
    )

    gr.Markdown("---")
    gr.Markdown("## 📄 Retrieved Context Chunks")
    chunks_output = gr.Textbox(
        label="Top-4 Retrieved Chunks (shared across all models)", lines=15,
    )

    gr.Markdown("---")
    gr.Markdown("## 📊 Evaluation Metrics")
    gr.Markdown(
        "All metrics are computed automatically from retrieved context — no ground truth input needed.\n\n"
        "| Metric | What it measures |\n"
        "|---|---|\n"
        "| **Faithfulness** | Cosine sim: answer vs retrieved context. Is the answer grounded in what was retrieved? |\n"
        "| **Semantic Sim** | Cosine sim: answer vs question. Is the answer on-topic? |\n"
        "| **Answer Relevancy** | Avg sentence-level cosine sim to question. Penalises filler sentences. |\n"
        "| **Citation Accuracy** | Fraction of top-chunk article numbers cited in the answer (auto ground truth). |\n"
        "| **Recall@4** | Fraction of expected articles found anywhere in the 4 retrieved chunks. |\n"
        "| **Precision@4** | Fraction of the 4 chunks with positive reranker score (genuinely relevant). |\n"
        "| **LLM-Judge** | 0–1 rubric score the model gives its own answer. |\n"
        "| **Latency (s)** | Total load + inference time per model. |"
    )

    metrics_output = gr.Markdown(label="Metric Comparison Table")
    chart_output   = gr.Plot(label="Visual Metric Comparison")

    outputs = [
        qwen_output, llama_output, gemma_output,
        chunks_output, latency_display, metrics_output, chart_output,
    ]

    submit_btn.click(fn=process_query, inputs=[question_input], outputs=outputs)
    question_input.submit(fn=process_query, inputs=[question_input], outputs=outputs)

    for btn, q in zip(example_btns, EXAMPLE_QUESTIONS):
        btn.click(
            fn=lambda _q=q: _q, inputs=[], outputs=[question_input],
        ).then(fn=process_query, inputs=[question_input], outputs=outputs)

demo.launch(ssr_mode=False)