"""Post-hoc analysis of results_n100: significance for every config, failure analysis, latency breakdown."""
import collections, json, sys, time, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tf_scp_experiment import TF_SCP_Pruner, build_samples, paired_bootstrap_p, mcnemar_exact_p

out = sys.argv[1] if len(sys.argv) > 1 else "results_n100"
rows = collections.defaultdict(dict)
for l in open(f"{out}/per_sample.jsonl"):
    r = json.loads(l); rows[r["config"]][r["id"]] = r
ids = list(rows["full"].keys())
full = rows["full"]

print("== Significance vs full (all configs) ==")
for k, d in rows.items():
    if k == "full": continue
    f1 = np.array([d[i]["f1"] for i in ids]); ff = np.array([full[i]["f1"] for i in ids])
    em = np.array([d[i]["em"] for i in ids]); fe = np.array([full[i]["em"] for i in ids])
    wins = int(((em > 0.5) & (fe < 0.5)).sum()); losses = int(((em < 0.5) & (fe > 0.5)).sum())
    print(f"{k:12s} dF1={100*(f1-ff).mean():+6.2f} p={paired_bootstrap_p(f1, ff):.3f} | EM wins/losses vs full={wins}/{losses} McNemar p={mcnemar_exact_p(em, fe):.3f}")

print("\n== Hybrid vs BM25-only (paired) ==")
h = rows["a0.5_b0.0"]; b = rows["a1.0_b0.0"]
hf = np.array([h[i]["f1"] for i in ids]); bf = np.array([b[i]["f1"] for i in ids])
print(f"dF1(hybrid-bm25)={100*(hf-bf).mean():+.2f} p={paired_bootstrap_p(hf, bf):.3f}")

print("\n== Answer-loss analysis ==")
for k in ("a1.0_b0.0", "a0.5_b0.0", "a0.0_b0.0", "a0.5_b0.5"):
    d = rows[k]
    lost = [i for i in ids if d[i]["retained"] < 0.5]
    kept = [i for i in ids if d[i]["retained"] > 0.5]
    lf = np.mean([d[i]["f1"] for i in lost]) * 100 if lost else float("nan")
    lff = np.mean([full[i]["f1"] for i in lost]) * 100 if lost else float("nan")
    kf = np.mean([d[i]["f1"] for i in kept]) * 100
    kff = np.mean([full[i]["f1"] for i in kept]) * 100
    print(f"{k:12s} lost={len(lost):2d} F1|lost={lf:5.1f} (full on same={lff:5.1f}) | kept={len(kept)} F1|kept={kf:5.1f} (full on same={kff:5.1f})")
lost_b = {i for i in ids if b[i]["retained"] < 0.5}; lost_h = {i for i in ids if h[i]["retained"] < 0.5}
print(f"BM25-lost ∩ hybrid-lost = {len(lost_b & lost_h)}; BM25-only lost & hybrid rescued = {len(lost_b - lost_h)}; hybrid-only lost = {len(lost_h - lost_b)}")

print("\n== Context stats ==")
print("sentences/query mean=%.1f sd=%.1f; full tokens mean=%.1f" % (
    np.mean([full[i]["n_sentences"] for i in ids]), np.std([full[i]["n_sentences"] for i in ids], ddof=1),
    np.mean([full[i]["full_tokens"] for i in ids])))

print("\n== Per-stage latency breakdown (ms, CPU, mean over 100 queries x 3 reps) ==")
samples, _ = build_samples(len(ids), 42, 2)
p = TF_SCP_Pruner()
for ex in samples[:5]:
    p.prune(ex["question"], ex["documents"], 0.5, 0.0)
seg, lex, sem, fuse = [], [], [], []
for _ in range(3):
    for ex in samples:
        t0 = time.perf_counter(); s, _ = p.segment(ex["documents"]); t1 = time.perf_counter()
        L = p.lexical_scores(ex["question"], s); t2 = time.perf_counter()
        S = p.semantic_scores(ex["question"], s); t3 = time.perf_counter()
        F = 0.5 * L + 0.5 * S; keep = np.flatnonzero(F >= F.mean()); t4 = time.perf_counter()
        seg.append(t1 - t0); lex.append(t2 - t1); sem.append(t3 - t2); fuse.append(t4 - t3)
for n, v in (("spaCy segmentation", seg), ("BM25 scoring", lex), ("MiniLM encoding+cosine", sem), ("fusion+threshold", fuse)):
    print(f"{n:24s} mean={1e3*np.mean(v):7.2f}  p50={1e3*np.median(v):7.2f}")
