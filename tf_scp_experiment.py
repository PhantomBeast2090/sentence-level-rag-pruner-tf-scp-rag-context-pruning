#!/usr/bin/env python3
"""
TF-SCP: Training-Free Sentence-Level Context Pruning for Retrieval-Augmented Generation
End-to-end experimental harness.

Pipeline per query
  1. Retrieval simulation (SQuAD v2, answerable subset): the gold paragraph plus
     k hard-negative paragraphs mined with paragraph-level BM25 from the same
     validation corpus, shuffled into a random order (fixed seed).
  2. Context pruning (TF-SCP): spaCy sentence segmentation -> per-sentence
     lexical (BM25Okapi) and semantic (all-MiniLM-L6-v2 cosine) scores, each
     min-max normalised within the query's candidate pool -> fusion
     F(s) = a*Lex + (1-a)*Sem -> keep sentences with F(s) >= mean + b*std.
  3. Generation with a local instruction-tuned LLM (greedy decoding).
  4. Metrics: token reduction (reader tokenizer), pruning latency, generation
     latency, EM, token-F1, ROUGE-L, gold-answer retention, paired significance
     tests against the full-context baseline.

Usage
  python tf_scp_experiment.py --n 100 --reader Qwen/Qwen2.5-0.5B-Instruct
"""
import argparse
import collections
import json
import math
import os
import platform
import random
import re
import string
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import spacy
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

TOKEN_RE = re.compile(r"\w+")


# ----------------------------------------------------------------------------
# TF-SCP pruner
# ----------------------------------------------------------------------------
@dataclass
class PruneResult:
    context: str
    n_sentences: int
    n_kept: int
    kept_idx: List[int]
    latency_ms: float
    threshold: float
    fused: np.ndarray = field(repr=False)


class TF_SCP_Pruner:
    """Training-free hybrid lexical/semantic sentence-level context pruner."""

    def __init__(self,
                 spacy_model: str = "en_core_web_sm",
                 embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu",
                 min_keep: int = 1):
        try:
            self.nlp = spacy.load(spacy_model, exclude=["ner", "lemmatizer", "tagger", "attribute_ruler"])
            self.segmenter = f"{spacy_model} (dependency parser)"
        except OSError:
            self.nlp = spacy.blank("en")
            self.nlp.add_pipe("sentencizer")
            self.segmenter = "spacy.blank('en') + sentencizer"
        self.stop_words = self.nlp.Defaults.stop_words
        self.encoder = SentenceTransformer(embed_model, device=device)
        self.min_keep = min_keep

    # -- stage 1: segmentation ------------------------------------------------
    def segment(self, documents: List[str]):
        sents, doc_ids = [], []
        for d_id, doc in enumerate(self.nlp.pipe(documents)):
            for s in doc.sents:
                t = s.text.strip()
                if t:
                    sents.append(t)
                    doc_ids.append(d_id)
        return sents, doc_ids

    # -- stage 2: scoring -----------------------------------------------------
    def _tokenize(self, text: str) -> List[str]:
        return [t for t in TOKEN_RE.findall(text.lower()) if t not in self.stop_words]

    @staticmethod
    def _minmax(x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        rng = x.max() - x.min()
        return np.zeros_like(x) if rng < 1e-12 else (x - x.min()) / rng

    def lexical_scores(self, query: str, sents: List[str]) -> np.ndarray:
        corpus = [self._tokenize(s) for s in sents]
        q = self._tokenize(query)
        if not q or sum(len(c) for c in corpus) == 0:
            return np.zeros(len(sents))
        return self._minmax(BM25Okapi(corpus).get_scores(q))

    def semantic_scores(self, query: str, sents: List[str]) -> np.ndarray:
        emb = self.encoder.encode([query] + sents, batch_size=64, convert_to_numpy=True,
                                  normalize_embeddings=True, show_progress_bar=False)
        return self._minmax(emb[1:] @ emb[0])

    # -- stages 3-4: fusion and dynamic thresholding ---------------------------
    def prune(self, query: str, documents: List[str], alpha: float = 0.5, beta: float = 0.0) -> PruneResult:
        t0 = time.perf_counter()
        sents, doc_ids = self.segment(documents)
        n = len(sents)
        if n == 0:
            return PruneResult("", 0, 0, [], (time.perf_counter() - t0) * 1e3, 0.0, np.zeros(0))
        lex = self.lexical_scores(query, sents) if alpha > 0.0 else np.zeros(n)
        sem = self.semantic_scores(query, sents) if alpha < 1.0 else np.zeros(n)
        fused = alpha * lex + (1.0 - alpha) * sem
        tau = float(fused.mean() + beta * fused.std())
        keep = np.flatnonzero(fused >= tau - 1e-12)
        if keep.size < self.min_keep:
            keep = np.sort(np.argsort(-fused)[: self.min_keep])
        # Re-assemble in original order, preserving document boundaries.
        by_doc: Dict[int, List[str]] = collections.OrderedDict()
        for i in keep:
            by_doc.setdefault(doc_ids[i], []).append(sents[i])
        context = "\n\n".join(" ".join(v) for v in by_doc.values())
        latency = (time.perf_counter() - t0) * 1e3
        return PruneResult(context, n, int(keep.size), keep.tolist(), latency, tau, fused)


# ----------------------------------------------------------------------------
# Reader LLM
# ----------------------------------------------------------------------------
class Reader:
    SYSTEM = "You are a precise extractive question-answering assistant."
    INSTR = ("Answer the question using only the context below. Reply with the shortest exact "
             "phrase from the context that answers the question, with no explanation.")

    def __init__(self, model_name: str, device: str, max_new_tokens: int = 32):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.seq2seq = bool(AutoConfig.from_pretrained(model_name).is_encoder_decoder)
        cls = AutoModelForSeq2SeqLM if self.seq2seq else AutoModelForCausalLM
        self.model = cls.from_pretrained(model_name, dtype=torch.float32).to(device).eval()
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def n_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def _prompt(self, context: str, question: str) -> str:
        user = f"{self.INSTR}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        if self.seq2seq or not getattr(self.tok, "chat_template", None):
            return user
        msgs = [{"role": "system", "content": self.SYSTEM}, {"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.inference_mode()
    def answer(self, context: str, question: str):
        prompt = self._prompt(context, question)
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        t0 = time.perf_counter()
        out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                  num_beams=1, pad_token_id=self.tok.pad_token_id)
        gen = out[0] if self.seq2seq else out[0, enc["input_ids"].shape[1]:]
        text = self.tok.decode(gen.cpu(), skip_special_tokens=True)
        latency = (time.perf_counter() - t0) * 1e3
        text = text.strip().split("\n")[0].strip()
        text = re.sub(r"^(answer\s*:\s*)", "", text, flags=re.I).strip()
        return text, latency, int(enc["input_ids"].shape[1])


# ----------------------------------------------------------------------------
# SQuAD metrics (official normalisation) + ROUGE-L
# ----------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: List[str]) -> float:
    return float(max(normalize_answer(pred) == normalize_answer(g) for g in golds))


def _f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    ns = sum(common.values())
    if not p or not g:
        return float(p == g)
    if ns == 0:
        return 0.0
    prec, rec = ns / len(p), ns / len(g)
    return 2 * prec * rec / (prec + rec)


def f1_score(pred: str, golds: List[str]) -> float:
    return max(_f1(pred, g) for g in golds)


def _lcs(a: List[str], b: List[str]) -> int:
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def rouge_l(pred: str, golds: List[str]) -> float:
    best = 0.0
    p = normalize_answer(pred).split()
    for g in golds:
        gt = normalize_answer(g).split()
        if not p or not gt:
            continue
        l = _lcs(p, gt)
        if l == 0:
            continue
        prec, rec = l / len(p), l / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def answer_retained(context: str, golds: List[str]) -> float:
    c = f" {normalize_answer(context)} "
    return float(any(f" {normalize_answer(g)} " in c for g in golds if normalize_answer(g)))


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def bootstrap_ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap_p(a, b, n_boot=10000, seed=0):
    """Two-sided p-value for H0: mean(a - b) = 0 (shifted-bootstrap)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if np.allclose(d, 0):
        return 1.0
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    centred = boots - d.mean()
    return float(min(1.0, (np.abs(centred) >= abs(d.mean())).mean() + 1.0 / n_boot))


def mcnemar_exact_p(a, b):
    a, b = np.asarray(a) > 0.5, np.asarray(b) > 0.5
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return float(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n))


# ----------------------------------------------------------------------------
# Data: SQuAD v2 with BM25 hard-negative "retrieved" documents
# ----------------------------------------------------------------------------
def build_samples(n: int, seed: int, num_distractors: int, split: str = "validation"):
    ds = load_dataset("rajpurkar/squad_v2", split=split)
    contexts = list(dict.fromkeys(ds["context"]))
    ctx_index = {c: i for i, c in enumerate(contexts)}
    tok = lambda t: TOKEN_RE.findall(t.lower())
    para_bm25 = BM25Okapi([tok(c) for c in contexts])

    answerable = [i for i, a in enumerate(ds["answers"]) if len(a["text"]) > 0]
    rng = random.Random(seed)
    rng.shuffle(answerable)
    samples = []
    for i in answerable[:n]:
        ex = ds[i]
        gold_id = ctx_index[ex["context"]]
        scores = para_bm25.get_scores(tok(ex["question"]))
        order = np.argsort(-scores)
        distractors = [int(j) for j in order if j != gold_id][:num_distractors]
        doc_ids = [gold_id] + distractors
        rng.shuffle(doc_ids)
        samples.append({
            "id": ex["id"], "question": ex["question"],
            "answers": list(dict.fromkeys(ex["answers"]["text"])),
            "documents": [contexts[j] for j in doc_ids],
            "gold_position": doc_ids.index(gold_id),
        })
    return samples, len(contexts)


# ----------------------------------------------------------------------------
# Experiment driver
# ----------------------------------------------------------------------------
def pick_device(pref: str) -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def summarise(rows: List[dict], full_rows: Optional[List[dict]]):
    g = lambda k: np.array([r[k] for r in rows], dtype=np.float64)
    s = {
        "ctx_tokens_mean": float(g("ctx_tokens").mean()),
        "token_reduction_mean": float(g("token_reduction").mean() * 100),
        "token_reduction_std": float(g("token_reduction").std(ddof=1) * 100),
        "sent_kept_pct": float((g("n_kept") / np.maximum(g("n_sentences"), 1)).mean() * 100),
        "prune_ms_mean": float(g("prune_ms").mean()),
        "prune_ms_p50": float(np.percentile(g("prune_ms"), 50)),
        "prune_ms_p95": float(np.percentile(g("prune_ms"), 95)),
        "gen_ms_mean": float(g("gen_ms").mean()),
        "e2e_ms_mean": float((g("gen_ms") + g("prune_ms")).mean()),
        "prompt_tokens_mean": float(g("prompt_tokens").mean()),
    }
    for m in ("em", "f1", "rougeL", "retained"):
        s[m] = float(g(m).mean() * 100)
        lo, hi = bootstrap_ci(g(m))
        s[f"{m}_ci"] = [lo * 100, hi * 100]
    if full_rows is not None:
        fe = np.array([r["em"] for r in full_rows]); ff = np.array([r["f1"] for r in full_rows])
        fr = np.array([r["rougeL"] for r in full_rows])
        s["delta_f1"] = float((g("f1") - ff).mean() * 100)
        s["p_f1"] = paired_bootstrap_p(g("f1"), ff)
        s["delta_em"] = float((g("em") - fe).mean() * 100)
        s["p_em_mcnemar"] = mcnemar_exact_p(g("em"), fe)
        s["f1_retention"] = float(g("f1").mean() / max(ff.mean(), 1e-9) * 100)
        s["rougeL_retention"] = float(g("rougeL").mean() / max(fr.mean(), 1e-9) * 100)
    return s


def print_tables(summary: Dict[str, dict], configs: List[dict], alphas, betas, beta_alpha):
    def row_name(c):
        return c["name"]
    print("\n" + "=" * 100)
    print("TABLE I: Token Reduction and Pruning Latency (mean over samples)")
    print("=" * 100)
    print("| Method | Ctx tokens | Token red. (%) | Sent. kept (%) | Prune ms (mean / p50 / p95) | Gen ms | E2E ms |")
    print("|---|---|---|---|---|---|---|")
    for c in configs[:3]:
        s = summary[c["key"]]
        print(f"| {row_name(c)} | {s['ctx_tokens_mean']:.1f} | {s['token_reduction_mean']:.2f} ± {s['token_reduction_std']:.2f} "
              f"| {s['sent_kept_pct']:.1f} | {s['prune_ms_mean']:.2f} / {s['prune_ms_p50']:.2f} / {s['prune_ms_p95']:.2f} "
              f"| {s['gen_ms_mean']:.1f} | {s['e2e_ms_mean']:.1f} |")

    print("\n" + "=" * 100)
    print("TABLE II: QA Performance (95% bootstrap CI in brackets; tests are paired vs. Standard RAG)")
    print("=" * 100)
    print("| Method | EM | F1 | ROUGE-L | Answer retained (%) | ΔF1 (p) | ΔEM (McNemar p) |")
    print("|---|---|---|---|---|---|---|")
    for c in configs[:3]:
        s = summary[c["key"]]
        d = "—" if "delta_f1" not in s else f"{s['delta_f1']:+.2f} ({s['p_f1']:.3f})"
        e = "—" if "delta_em" not in s else f"{s['delta_em']:+.2f} ({s['p_em_mcnemar']:.3f})"
        print(f"| {row_name(c)} | {s['em']:.2f} [{s['em_ci'][0]:.1f}, {s['em_ci'][1]:.1f}] "
              f"| {s['f1']:.2f} [{s['f1_ci'][0]:.1f}, {s['f1_ci'][1]:.1f}] "
              f"| {s['rougeL']:.2f} [{s['rougeL_ci'][0]:.1f}, {s['rougeL_ci'][1]:.1f}] "
              f"| {s['retained']:.1f} | {d} | {e} |")

    for title, keys in (("TABLE III: Ablation over fusion weight α (β = 0.0)", [f"a{a}_b0.0" for a in alphas]),
                        (f"TABLE IV: Sensitivity to threshold β (α = {beta_alpha})", [f"a{beta_alpha}_b{b}" for b in betas])):
        print("\n" + "=" * 100)
        print(title)
        print("=" * 100)
        print("| α | β | Token red. (%) | Prune ms | Answer retained (%) | EM | F1 | ROUGE-L | F1 retention (%) |")
        print("|---|---|---|---|---|---|---|---|---|")
        for k in keys:
            s = summary[k]
            a, b = k[1:].split("_b")
            print(f"| {float(a):.2f} | {float(b):+.2f} | {s['token_reduction_mean']:.2f} | {s['prune_ms_mean']:.2f} "
                  f"| {s['retained']:.1f} | {s['em']:.2f} | {s['f1']:.2f} | {s['rougeL']:.2f} | {s['f1_retention']:.1f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--distractors", type=int, default=2, help="BM25 hard negatives per query")
    ap.add_argument("--reader", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    ap.add_argument("--reader-device", default="auto")
    ap.add_argument("--pruner-device", default="cpu")
    ap.add_argument("--alphas", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--betas", default="-0.5,0.0,0.5")
    ap.add_argument("--beta-alpha", type=float, default=0.5)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    os.makedirs(args.out, exist_ok=True)
    reader_device = pick_device(args.reader_device)
    alphas = [float(a) for a in args.alphas.split(",")]
    betas = [float(b) for b in args.betas.split(",")]

    configs = [
        {"key": "full", "name": "Standard RAG (full context)", "alpha": None, "beta": None},
        {"key": "a1.0_b0.0", "name": "BM25-only pruning (α=1.0, β=0.0)", "alpha": 1.0, "beta": 0.0},
        {"key": "a0.5_b0.0", "name": "TF-SCP hybrid (α=0.5, β=0.0)", "alpha": 0.5, "beta": 0.0},
    ]
    seen = {c["key"] for c in configs}
    for a in alphas:
        for b in [0.0]:
            k = f"a{a}_b{b}"
            if k not in seen:
                configs.append({"key": k, "name": f"α={a}, β={b}", "alpha": a, "beta": b}); seen.add(k)
    for b in betas:
        k = f"a{args.beta_alpha}_b{b}"
        if k not in seen:
            configs.append({"key": k, "name": f"α={args.beta_alpha}, β={b}", "alpha": args.beta_alpha, "beta": b}); seen.add(k)

    print(f"[env] python={platform.python_version()} torch={torch.__version__} transformers={transformers.__version__} "
          f"spacy={spacy.__version__} platform={platform.platform()} processor={platform.processor()}")
    print(f"[env] reader={args.reader} on {reader_device} | pruner encoder={args.embed_model} on {args.pruner_device}")

    t = time.time()
    samples, corpus_size = build_samples(args.n, args.seed, args.distractors)
    print(f"[data] squad_v2/validation: {len(samples)} answerable queries, corpus={corpus_size} paragraphs, "
          f"docs/query={args.distractors + 1} ({time.time() - t:.1f}s)")

    pruner = TF_SCP_Pruner(args.spacy_model, args.embed_model, args.pruner_device)
    print(f"[pruner] segmenter={pruner.segmenter}")
    reader = Reader(args.reader, reader_device)

    # Warm-up (excluded from timing)
    for _ in range(3):
        for a in (0.0, 0.5, 1.0):
            pruner.prune(samples[0]["question"], samples[0]["documents"], a, 0.0)
    reader.answer(samples[0]["documents"][0], samples[0]["question"])

    per_cfg: Dict[str, List[dict]] = {c["key"]: [] for c in configs}
    per_sample_path = os.path.join(args.out, "per_sample.jsonl")
    t_run = time.time()
    with open(per_sample_path, "w") as fout:
        for si, ex in enumerate(samples):
            full_ctx = "\n\n".join(ex["documents"])
            full_tokens = reader.n_tokens(full_ctx)
            n_full_sents = len(pruner.segment(ex["documents"])[0])
            for c in configs:
                if c["alpha"] is None:
                    ctx, n_s, n_k, p_ms = full_ctx, n_full_sents, n_full_sents, 0.0
                else:
                    pr = pruner.prune(ex["question"], ex["documents"], c["alpha"], c["beta"])
                    ctx, n_s, n_k, p_ms = pr.context, pr.n_sentences, pr.n_kept, pr.latency_ms
                pred, g_ms, p_tok = reader.answer(ctx, ex["question"])
                ctx_tokens = reader.n_tokens(ctx)
                row = {
                    "id": ex["id"], "config": c["key"], "pred": pred, "answers": ex["answers"],
                    "ctx_tokens": ctx_tokens, "full_tokens": full_tokens,
                    "token_reduction": 1.0 - ctx_tokens / max(full_tokens, 1),
                    "n_sentences": n_s, "n_kept": n_k, "prune_ms": p_ms, "gen_ms": g_ms,
                    "prompt_tokens": p_tok,
                    "em": exact_match(pred, ex["answers"]), "f1": f1_score(pred, ex["answers"]),
                    "rougeL": rouge_l(pred, ex["answers"]), "retained": answer_retained(ctx, ex["answers"]),
                }
                per_cfg[c["key"]].append(row)
                fout.write(json.dumps(row) + "\n")
            if (si + 1) % 10 == 0 or si == 0:
                el = time.time() - t_run
                f_em = np.mean([r["em"] for r in per_cfg["full"]]) * 100
                h_em = np.mean([r["em"] for r in per_cfg["a0.5_b0.0"]]) * 100
                print(f"[run] {si + 1:>4}/{len(samples)}  elapsed={el:6.1f}s  running EM full={f_em:.1f} hybrid={h_em:.1f}")

    summary = {c["key"]: summarise(per_cfg[c["key"]], None if c["key"] == "full" else per_cfg["full"])
               for c in configs}
    print_tables(summary, configs, alphas, betas, args.beta_alpha)
    meta = {"args": vars(args), "reader_device": reader_device, "n_samples": len(samples),
            "corpus_size": corpus_size, "segmenter": pruner.segmenter,
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "transformers": transformers.__version__, "spacy": spacy.__version__},
            "configs": configs, "summary": summary}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n[done] total {time.time() - t_run:.1f}s | wrote {args.out}/summary.json and {per_sample_path}")


if __name__ == "__main__":
    main()
