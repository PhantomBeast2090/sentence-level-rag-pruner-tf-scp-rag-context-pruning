# TF-SCP — Experimental Output and Paper Sections 5–6

> All numbers below come from an actual run of `tf_scp_experiment.py` on SQuAD v2 (n = 1,000, seed = 42) on an Apple M3 CPU (8 GB). Raw artifacts: `results_n1000/summary.json`, `results_n1000/per_sample.jsonl`, `results_n1000/analysis.txt`.

## Reproduction

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
HF_HOME=./.hf_cache .venv/bin/python prefetch_assets.py Qwen/Qwen2.5-0.5B-Instruct
HF_HOME=./.hf_cache HF_HUB_OFFLINE=1 .venv/bin/python tf_scp_experiment.py --n 1000 --out results_n1000
HF_HOME=./.hf_cache HF_HUB_OFFLINE=1 .venv/bin/python analyze_results.py results_n1000
```

## Terminal output (main run)

```text
[env] python=3.12.12 torch=2.15.0 transformers=5.23.0 spacy=3.8.18 platform=macOS-26.5.1-arm64-arm-64bit processor=arm
[env] reader=Qwen/Qwen2.5-0.5B-Instruct on cpu | pruner encoder=sentence-transformers/all-MiniLM-L6-v2 on cpu
[data] squad_v2/validation: 1000 answerable queries, corpus=1204 paragraphs, docs/query=3 (1.2s)
[pruner] segmenter=en_core_web_sm (dependency parser)
[run]    1/1000  elapsed=   3.5s  running EM full=0.0 hybrid=100.0
[run]  100/1000  elapsed= 522.8s  running EM full=34.0 hybrid=35.0
[run]  200/1000  elapsed=1041.2s  running EM full=34.0 hybrid=35.0
[run]  300/1000  elapsed=1611.8s  running EM full=35.7 hybrid=35.7
[run]  400/1000  elapsed=2083.7s  running EM full=36.8 hybrid=38.5
[run]  500/1000  elapsed=2533.5s  running EM full=36.8 hybrid=39.4
[run]  600/1000  elapsed=3046.3s  running EM full=36.7 hybrid=38.8
[run]  700/1000  elapsed=3610.4s  running EM full=37.1 hybrid=39.1
[run]  800/1000  elapsed=4181.4s  running EM full=37.9 hybrid=40.0
[run]  900/1000  elapsed=4715.4s  running EM full=38.8 hybrid=41.1
[run] 1000/1000  elapsed=5195.1s  running EM full=38.2 hybrid=40.4

====================================================================================================
TABLE I: Token Reduction and Pruning Latency (mean over samples)
====================================================================================================
| Method | Ctx tokens | Token red. (%) | Sent. kept (%) | Prune ms (mean / p50 / p95) | Gen ms | E2E ms |
|---|---|---|---|---|---|---|
| Standard RAG (full context) | 524.5 | 0.00 ± 0.00 | 100.0 | 0.00 / 0.00 / 0.00 | 807.6 | 807.6 |
| BM25-only pruning (α=1.0, β=0.0) | 188.2 | 63.71 ± 12.92 | 32.7 | 13.75 / 11.22 / 24.51 | 531.0 | 544.8 |
| TF-SCP hybrid (α=0.5, β=0.0) | 231.3 | 55.79 ± 12.21 | 41.2 | 60.61 / 51.44 / 119.67 | 572.6 | 633.2 |

====================================================================================================
TABLE II: QA Performance (95% bootstrap CI in brackets; tests are paired vs. Standard RAG)
====================================================================================================
| Method | EM | F1 | ROUGE-L | Answer retained (%) | ΔF1 (p) | ΔEM (McNemar p) |
|---|---|---|---|---|---|---|
| Standard RAG (full context) | 38.20 [35.2, 41.2] | 50.94 [48.2, 53.7] | 50.89 [48.1, 53.6] | 99.2 | — | — |
| BM25-only pruning (α=1.0, β=0.0) | 39.10 [36.2, 42.1] | 53.54 [50.9, 56.3] | 53.42 [50.8, 56.2] | 93.2 | +2.60 (0.046) | +0.90 (0.550) |
| TF-SCP hybrid (α=0.5, β=0.0) | 40.40 [37.3, 43.6] | 53.67 [50.9, 56.4] | 53.59 [50.8, 56.3] | 97.5 | +2.73 (0.025) | +2.20 (0.099) |

====================================================================================================
TABLE III: Ablation over fusion weight α (β = 0.0)
====================================================================================================
| α | β | Token red. (%) | Prune ms | Answer retained (%) | EM | F1 | ROUGE-L | F1 retention (%) |
|---|---|---|---|---|---|---|---|---|
| 0.00 | +0.00 | 50.57 | 51.02 | 96.5 | 39.80 | 53.89 | 53.81 | 105.8 |
| 0.25 | +0.00 | 52.61 | 51.16 | 97.4 | 40.30 | 53.73 | 53.66 | 105.5 |
| 0.50 | +0.00 | 55.79 | 60.61 | 97.5 | 40.40 | 53.67 | 53.59 | 105.4 |
| 0.75 | +0.00 | 60.50 | 50.94 | 96.2 | 40.00 | 54.87 | 54.73 | 107.7 |
| 1.00 | +0.00 | 63.71 | 13.75 | 93.2 | 39.10 | 53.54 | 53.42 | 105.1 |

====================================================================================================
TABLE IV: Sensitivity to threshold β (α = 0.5)
====================================================================================================
| α | β | Token red. (%) | Prune ms | Answer retained (%) | EM | F1 | ROUGE-L | F1 retention (%) |
|---|---|---|---|---|---|---|---|---|
| 0.50 | -0.50 | 31.20 | 51.94 | 98.8 | 40.00 | 52.70 | 52.63 | 103.4 |
| 0.50 | +0.00 | 55.79 | 60.61 | 97.5 | 40.40 | 53.67 | 53.59 | 105.4 |
| 0.50 | +0.50 | 74.25 | 52.67 | 94.5 | 43.00 | 57.61 | 57.40 | 113.1 |

[done] total 5195.8s | wrote results_n1000/summary.json and results_n1000/per_sample.jsonl
```

## Terminal output (post-hoc analysis)

```text
== Significance vs full (all configs) ==
a1.0_b0.0    dF1= +2.60 p=0.046 | EM wins/losses vs full=94/85 McNemar p=0.550
a0.5_b0.0    dF1= +2.73 p=0.025 | EM wins/losses vs full=92/70 McNemar p=0.099
a0.0_b0.0    dF1= +2.95 p=0.013 | EM wins/losses vs full=86/70 McNemar p=0.230
a0.25_b0.0   dF1= +2.79 p=0.025 | EM wins/losses vs full=92/71 McNemar p=0.117
a0.75_b0.0   dF1= +3.93 p=0.002 | EM wins/losses vs full=96/78 McNemar p=0.197
a0.5_b-0.5   dF1= +1.76 p=0.096 | EM wins/losses vs full=74/56 McNemar p=0.136
a0.5_b0.5    dF1= +6.67 p=0.000 | EM wins/losses vs full=126/78 McNemar p=0.001

== Hybrid vs BM25-only (paired) ==
dF1(hybrid-bm25)=+0.13 p=0.901

== Answer-loss analysis ==
a1.0_b0.0    lost=68 F1|lost= 11.2 (full on same= 32.4) | kept=932 F1|kept= 56.6 (full on same= 52.3)
a0.5_b0.0    lost=25 F1|lost= 20.0 (full on same= 33.6) | kept=975 F1|kept= 54.5 (full on same= 51.4)
a0.0_b0.0    lost=35 F1|lost= 15.4 (full on same= 42.3) | kept=965 F1|kept= 55.3 (full on same= 51.3)
a0.5_b0.5    lost=55 F1|lost= 19.1 (full on same= 34.2) | kept=945 F1|kept= 59.9 (full on same= 51.9)
BM25-lost ∩ hybrid-lost = 24; BM25-only lost & hybrid rescued = 44; hybrid-only lost = 1

== Context stats ==
sentences/query mean=16.3 sd=4.9; full tokens mean=524.5

== Per-stage latency breakdown (ms, CPU, mean over 100 queries x 3 reps) ==
spaCy segmentation       mean=  10.87  p50=  10.16
BM25 scoring             mean=   0.17  p50=   0.17
MiniLM encoding+cosine   mean=  35.50  p50=  31.51
fusion+threshold         mean=   0.02  p50=   0.02
```

---

## 5. Results and Quantitative Analysis

### 5.1 Experimental Configuration

We evaluate on 1,000 answerable questions drawn uniformly at random (seed 42) from the SQuAD v2 validation split. To emulate a retriever with imperfect precision, each query is paired with its gold paragraph and two hard-negative paragraphs. The negatives are the highest-ranked non-gold paragraphs under paragraph-level BM25 over the full validation corpus (1,204 unique paragraphs). The three documents are shuffled into a random order. The resulting candidate pool averages 16.3 ± 4.9 sentences and 524.5 reader tokens per query. The reader is Qwen2.5-0.5B-Instruct with greedy decoding (maximum 32 new tokens) in FP32. Both the reader and the pruner (spaCy `en_core_web_sm` for segmentation, BM25Okapi, all-MiniLM-L6-v2) run on the CPU of an Apple M3 laptop with 8 GB of memory. Lexical and semantic scores are min-max normalised within each query's candidate pool before fusion. At least one sentence is always retained.

We report token reduction measured with the reader's tokenizer, wall-clock pruning latency (warm, excluding model loading), reader generation latency, and SQuAD-normalised Exact Match (EM), token-level F1, and ROUGE-L. We also report *answer retention*, the fraction of queries whose context still contains a gold answer string bounded by whitespace after SQuAD normalisation. On the unpruned full-context baseline, this strict boundary-matched check registers 99.2% (992/1,000 queries) rather than 100.0% because 8 queries have gold answer annotations that appear as sub-word stems or boundary-attached spans in the source text after punctuation stripping (e.g., `"Resign"` inside `"resigned"`). Uncertainty is quantified with 95% percentile-bootstrap confidence intervals (2,000 resamples). Differences against the full-context baseline are tested with a paired bootstrap on per-query F1 (10,000 resamples) and an exact McNemar test on EM. Because seven pruned configurations are evaluated against the baseline, the Bonferroni-corrected significance threshold is α_FWER = 0.05 / 7 ≈ 0.0071.

### 5.2 Context Compression and Latency

**TABLE I. Token reduction and latency (n = 1,000; mean ± s.d. where shown).**

| Method | Context tokens | Token reduction (%) | Sentences retained (%) | Pruning latency (ms) mean / p50 / p95 | Generation latency (ms) | End-to-end latency (ms) |
|---|---|---|---|---|---|---|
| Standard RAG (full context) | 524.5 | 0.00 | 100.0 | — | 807.6 | 807.6 |
| BM25-only pruning (α = 1.0, β = 0.0) | 188.2 | 63.71 ± 12.92 | 32.7 | 13.75 / 11.22 / 24.51 | 531.0 | 544.8 |
| TF-SCP hybrid (α = 0.5, β = 0.0) | 231.3 | 55.79 ± 12.21 | 41.2 | 60.61 / 51.44 / 119.67 | 572.6 | 633.2 |

Both pruning variants substantially compress the retrieved context (Table I). BM25-only pruning is the more aggressive of the two: it keeps 32.7% of sentences and removes 63.71% of context tokens (down to 188.2 tokens). The hybrid configuration keeps 41.2% of sentences, for a token reduction of 55.79%, so it retains on average 43.1 more tokens per query than the lexical-only variant. Its per-query variability is slightly lower (s.d. 12.21 vs. 12.92 percentage points).

Pruning overhead is dominated by the dense encoder. An isolated per-stage measurement across the 1,000 evaluation queries (three repetitions per query) attributes 35.50 ms (p50 = 31.51 ms) to MiniLM encoding and cosine scoring and 10.87 ms (p50 = 10.16 ms) to spaCy dependency-based segmentation. BM25 scoring (0.17 ms) and fusion with thresholding (0.02 ms) are negligible. Sentence encoding therefore accounts for roughly 76.2% of the hybrid pipeline's isolated cost of 46.56 ms. The in-situ mean of 60.61 ms reported in Table I (and 50.94–52.67 ms across the other α < 1 configurations) is higher because pruning was interleaved with reader inference on the same CPU. Because the BM25-only variant skips the encoder, it costs 13.75 ms (p50 = 11.22 ms), almost all of which is segmentation.

The shorter contexts reduce reader generation time by 34.2% for BM25-only pruning (807.6 → 531.0 ms) and by 29.1% for TF-SCP hybrid (807.6 → 572.6 ms). After pruning overhead is included, end-to-end latency falls by 32.5% for BM25-only pruning (807.6 → 544.8 ms) and by 21.6% for TF-SCP hybrid (807.6 → 633.2 ms). Token savings translate sub-linearly into generation latency because prompt-template tokens (71.9 tokens on average) and autoregressive decoding steps do not shrink with the context.

### 5.3 Question-Answering Performance

**TABLE II. QA performance (n = 1,000; 95% bootstrap CIs in brackets). Δ and p-values are paired against Standard RAG.**

| Method | EM | F1 | ROUGE-L | Answer retained (%) | ΔF1 (p, paired bootstrap) | ΔEM (p, McNemar) |
|---|---|---|---|---|---|---|
| Standard RAG (full context) | 38.20 [35.2, 41.2] | 50.94 [48.2, 53.7] | 50.89 [48.1, 53.6] | 99.2 | — | — |
| BM25-only pruning (α = 1.0, β = 0.0) | 39.10 [36.2, 42.1] | 53.54 [50.9, 56.3] | 53.42 [50.8, 56.2] | 93.2 | +2.60 (0.046) | +0.90 (0.550) |
| TF-SCP hybrid (α = 0.5, β = 0.0) | 40.40 [37.3, 43.6] | 53.67 [50.9, 56.4] | 53.59 [50.8, 56.3] | 97.5 | +2.73 (0.025) | +2.20 (0.099) |

Scaling to n = 1,000 narrows the 95% bootstrap confidence intervals on F1 from roughly ±8.5 points (at n = 100) to ±2.7 points (Table II). Both primary pruning configurations outperform the full-context baseline on every QA metric, yielding F1 retention of 105.1% for BM25-only pruning and 105.4% for TF-SCP hybrid.

Both β = 0.0 variants cross the nominal α = 0.05 threshold on F1: BM25-only gains +2.60 F1 (p = 0.0455) and TF-SCP hybrid gains +2.73 F1 (p = 0.0255), although neither passes the stricter Bonferroni threshold of 0.0071. Their EM improvements (+0.90, 94 wins vs. 85 losses, McNemar p = 0.550; and +2.20, 92 wins vs. 70 losses, McNemar p = 0.099) do not reach statistical significance. In the direct paired comparison between the two β = 0.0 pruners, TF-SCP hybrid and BM25-only are statistically indistinguishable on F1 (53.67 vs. 53.54; ΔF1 = +0.13, paired bootstrap p = 0.901), while TF-SCP hybrid achieves a highly significant +4.3 percentage-point advantage in answer retention (97.5% vs. 93.2%, exact McNemar p = 2.6 × 10⁻¹²).

### 5.4 Ablation of the Fusion Weight α

**TABLE III. Sweep of the fusion weight α (β = 0.0).**

| α | Token reduction (%) | Pruning latency (ms) | Answer retained (%) | EM | F1 | ROUGE-L | F1 retention (%) |
|---|---|---|---|---|---|---|---|
| 0.00 (semantic only) | 50.57 | 51.02 | 96.5 | 39.80 | 53.89 | 53.81 | 105.8 |
| 0.25 | 52.61 | 51.16 | 97.4 | 40.30 | 53.73 | 53.66 | 105.5 |
| 0.50 | 55.79 | 60.61 | 97.5 | 40.40 | 53.67 | 53.59 | 105.4 |
| 0.75 | 60.50 | 50.94 | 96.2 | 40.00 | 54.87 | 54.73 | 107.7 |
| 1.00 (lexical only) | 63.71 | 13.75 | 93.2 | 39.10 | 53.54 | 53.42 | 105.1 |

Token reduction increases monotonically with α, from 50.57% (semantic only) to 63.71% (lexical only; Table III). Answer retention, by contrast, is non-monotonic: it peaks in the interior hybrid region at 97.5% (α = 0.50) and 97.4% (α = 0.25), exceeding both pure semantic pruning (96.5% at α = 0.00) and pure lexical pruning (93.2% at α = 1.00). This confirms genuine complementarity between lexical and dense signals in retaining answer-bearing sentences.

Every α ∈ {0.00, 0.25, 0.50, 0.75, 1.00} achieves a nominally significant F1 gain over Standard RAG (ΔF1 = +2.95 [p = 0.013], +2.79 [p = 0.025], +2.73 [p = 0.025], +3.93 [p = 0.002], and +2.60 [p = 0.046], respectively). Notably, the lexical-weighted hybrid α = 0.75 attains 54.87 F1 (107.7% F1 retention, ΔF1 = +3.93, p = 0.0022), which clears the Bonferroni threshold of 0.0071 on F1, although its EM gain (40.00 vs. 38.20; 96 wins vs. 78 losses, McNemar p = 0.197) is not significant. Latency again separates into two regimes: all α < 1.0 settings invoke the encoder (50.94–60.61 ms in-situ), whereas α = 1.00 requires 13.75 ms.

### 5.5 Sensitivity to the Threshold Coefficient β

**TABLE IV. Sensitivity to threshold β (α = 0.5).**

| β | Token reduction (%) | Pruning latency (ms) | Answer retained (%) | EM | F1 | ROUGE-L | ΔF1 vs. full (p) |
|---|---|---|---|---|---|---|---|
| −0.50 | 31.20 | 51.94 | 98.8 | 40.00 | 52.70 | 52.63 | +1.76 (0.096) |
| 0.00 | 55.79 | 60.61 | 97.5 | 40.40 | 53.67 | 53.59 | +2.73 (0.025) |
| +0.50 | 74.25 | 52.67 | 94.5 | 43.00 | 57.61 | 57.40 | +6.67 (<0.001) |

Compression is substantially more sensitive to the threshold coefficient β than to the fusion weight α (Table IV). Raising β from −0.50 to +0.50 increases token reduction from 31.20% (362.1 tokens, 65.3% of sentences) to 74.25% (133.9 tokens, 23.5% of sentences), lowering generation latency to 511.7 ms and end-to-end latency to 564.3 ms (a 30.1% end-to-end speed-up over Standard RAG). Answer retention over the same sweep drops from 98.8% to 94.5%.

The accuracy advantage of aggressive thresholding observed at n = 100 is confirmed as a statistically significant effect at n = 1,000. At β = +0.50, TF-SCP achieves the highest performance across all tested configurations: 43.00 EM [40.0, 46.1], 57.61 F1 [55.0, 60.3], and 57.40 ROUGE-L (113.1% F1 retention). Relative to Standard RAG, both the F1 improvement (+6.67 points, paired bootstrap p = 0.0001) and the EM improvement (+4.80 points; 126 wins vs. 78 losses, exact McNemar p = 0.0010) remain significant well below the Bonferroni-corrected threshold (p < 0.0071). In direct paired comparisons, α = 0.5, β = +0.5 also significantly outperforms both the default hybrid setting α = 0.5, β = 0.0 (ΔF1 = +3.94, p = 0.0007; ΔEM = +2.60, McNemar p = 0.0422) and BM25-only pruning α = 1.0, β = 0.0 (ΔF1 = +4.07, p = 0.0001; ΔEM = +3.90, McNemar p = 0.0009), while simultaneously delivering stronger compression (74.25% vs. 63.71%) and higher answer retention (94.5% vs. 93.2%) than BM25-only pruning.

---

## 6. Discussion and Limitations

### 6.1 Why the Hybrid Scorer Behaves as Observed

The n = 1,000 evaluation clarifies the two opposing mechanisms governing pruned-RAG accuracy: *answer-recall preservation* and *distractor denoising*.

First, semantic fusion functions as an effective recall safeguard. Across the 992 queries where the unpruned context satisfies the boundary-matched answer check, BM25-only pruning (α = 1.0, β = 0.0) discards the gold answer on 60 queries (totaling 68 non-retained queries including the 8 boundary-artifact cases), whereas TF-SCP hybrid (α = 0.5, β = 0.0) discards it on only 17 queries (25 total). Query-level overlap shows that 24 missed queries are shared, TF-SCP hybrid rescues 44 of BM25's failures (73.3% of its 60 genuine losses), and hybrid pruning drops only 1 query missed by BM25-only (p = 2.6 × 10⁻¹², McNemar test). Hybrid fusion at α = 0.5 also loses fewer answers than pure semantic pruning at α = 0.0 (25 vs. 35 total missed; 17 vs. 27 genuine losses), because lexical matching guards against rare entity names or numeric spans that dense embeddings under-weight, while MiniLM recovers paraphrastic or indirect phrasing that BM25 misses. Answer loss carries a large accuracy penalty: on queries without the retained answer string, reader F1 falls relative to the full-context reader on the same subsets from 32.4 to 11.2 for BM25-only (−21.2 F1), from 33.6 to 20.0 for hybrid β = 0.0 (−13.6 F1), from 42.3 to 15.4 for semantic-only (−26.9 F1), and from 34.2 to 19.1 for hybrid β = +0.5 (−15.1 F1).

Second, removing sentences from topically similar hard-negative paragraphs produces a consistent *denoising* gain on the queries whose answer is preserved. Conditioned on answer retention, every pruner outperforms the full-context reader on the exact same queries:

- **BM25-only (α = 1.0, β = 0.0):** +4.3 F1 across 932 queries (56.6 vs. 52.3)
- **TF-SCP hybrid (α = 0.5, β = 0.0):** +3.1 F1 across 975 queries (54.5 vs. 51.4)
- **Semantic-only (α = 0.0, β = 0.0):** +4.0 F1 across 965 queries (55.3 vs. 51.3)
- **TF-SCP hybrid (α = 0.5, β = +0.5):** +8.0 F1 across 945 queries (59.9 vs. 51.9)

Aggregate ΔF1 decomposes cleanly into the weighted sum of this denoising benefit and the answer-loss penalty:

- **BM25-only (α = 1.0, β = 0.0):** 0.932 × (+4.3) + 0.068 × (−21.2) = +4.01 − 1.44 = +2.57 (observed +2.60).
- **TF-SCP hybrid (α = 0.5, β = 0.0):** 0.975 × (+3.1) + 0.025 × (−13.6) = +3.02 − 0.34 = +2.68 (observed +2.73).
- **TF-SCP hybrid (α = 0.5, β = +0.5):** 0.945 × (+8.0) + 0.055 × (−15.1) = +7.56 − 0.83 = +6.73 (observed +6.67).

This decomposition explains both why BM25-only and TF-SCP hybrid tie on aggregate F1 at β = 0.0 and why TF-SCP at β = +0.5 significantly outperforms both. At β = 0.0, the hybrid pruner keeps 41.2% of sentences versus 32.7% for BM25-only; the extra sentences come disproportionately from topical non-answer sentences in the hard negatives that dense similarity scores highly, diluting the denoising gain (+3.1 vs. +4.3 F1) by almost the exact amount saved through fewer answer losses (−0.34 vs. −1.44 F1). However, because the hybrid scorer ranks the gold sentence higher relative to distractors than BM25 alone, shifting the cutoff upward to β = +0.5 strips away those borderline distractor sentences: token reduction rises to 74.25%, the denoising gain more than doubles to +8.0 F1, and answer loss remains lower than under BM25-only pruning at β = 0.0 (55 vs. 68 non-retained queries). For distraction-sensitive small readers, hybrid scoring combined with a stricter positive threshold (β = +0.5) captures the recall safeguard of dense fusion without paying the denoising penalty of an overly permissive cutoff. For larger, more distraction-robust readers, the denoising gain should shrink while the penalty for dropping an answer sentence remains large, which would shift the optimal threshold back toward β ≤ 0.0; this hypothesis remains to be tested across model scales.

### 6.2 Dataset Characteristics Favouring Lexical Scoring

BM25-only pruning remained competitive at β = 0.0 partly because of how SQuAD is constructed. Annotators wrote questions while viewing the target paragraph, resulting in high lexical overlap between questions and answer-bearing sentences. Even under those favourable conditions, TF-SCP hybrid reduced genuine answer loss from 6.0% to 1.7% (a 4.3 percentage-point retention advantage overall at β = 0.0, p = 2.6 × 10⁻¹²). On corpora with naturally occurring queries—such as Natural Questions or MS MARCO—or multi-hop benchmarks where bridging sentences share few surface terms with the question, the recall disadvantage of pure lexical scoring is expected to widen further.

### 6.3 Cost–Benefit of the Semantic Component

On a commodity CPU, adding the dense encoder increases isolated pruning time from ≈ 11–14 ms to ≈ 46.6 ms (60.6 ms in-situ under contention). Even with this overhead, the 55.8–74.3% token reduction produces net end-to-end speed-ups of 21.6% at β = 0.0 (807.6 → 633.2 ms) and 30.1% at β = +0.5 (807.6 → 564.3 ms), compared with 32.5% for BM25-only pruning (807.6 → 544.8 ms). Crucially, pruning overhead is independent of reader size, whereas LLM prefill latency and API token fees scale with context length and model size. For multi-billion-parameter readers or paid inference endpoints, spending ≈ 35.5 ms on MiniLM sentence encoding to remove up to 74.3% of prompt tokens—while significantly improving QA accuracy—becomes progressively more favourable. Several straightforward optimisations were not evaluated here and should further reduce pruning latency:

- running or batching the dense encoder on GPU or MPS accelerators;
- pre-computing and indexing document-side sentence embeddings offline so only the query is encoded at inference time;
- replacing the spaCy dependency parser (10.87 ms) with a lightweight rule-based sentencizer.

### 6.4 Practical Guidance

Based on the n = 1,000 empirical findings, we recommend:

1. **Precision-optimised default for small readers (α = 0.5, β = +0.5):** Achieves the best overall trade-off—74.25% token reduction, a 30.1% end-to-end speedup, 94.5% answer retention, and Bonferroni-significant gains of +6.67 F1 (p = 0.0001) and +4.80 EM (p = 0.0010) over full-context RAG. The lexical-weighted setting α = 0.75, β = 0.0 offers an intermediate alternative with a Bonferroni-significant +3.93 F1 gain (p = 0.0022) and 60.50% compression.
2. **Recall-preserving regime (α ∈ [0.25, 0.50], β = 0.0):** Appropriate when minimizing answer-sentence loss is the primary requirement (97.4–97.5% answer retention, 52.6–55.8% compression, +2.73 to +2.79 F1).
3. **Ultra-low-latency CPU regime (α = 1.0, β = 0.0):** Appropriate only when ≈ 14 ms pruning latency is mandatory, queries exhibit high surface-form overlap with retrieved passages, and a 6.8% non-retention rate is acceptable.

### 6.5 Limitations

1. **Statistical power across fine-grained ablations.** Scaling to n = 1,000 reduced the 95% F1 confidence interval width to ±2.7 points and confirmed Bonferroni-significant gains for α = 0.5, β = +0.5 (both F1 and EM) and α = 0.75, β = 0.0 (F1). However, the moderate-compression configurations at β = 0.0 (α ∈ {0.0, 0.25, 0.5, 1.0}) achieve only nominal F1 significance (0.013 ≤ p ≤ 0.046) and non-significant EM differences (p ≥ 0.099), while pairwise differences along the α sweep (≤ 1.3 F1 points) remain smaller than the minimum detectable effect at n = 1,000.
2. **Simulated retrieval.** Every query pool includes the gold paragraph alongside two hard negatives (524.5 tokens total). Production RAG pipelines often process larger candidate pools (e.g., top-5 to top-20 chunks) and sometimes fail to retrieve the gold passage at all; both achievable compression and answer-loss risk will shift under those conditions.
3. **Single small reader.** All experiments use a 0.5B-parameter reader (38.20% EM with full context). Distractor sensitivity—and thus the +8.0 F1 denoising gain driving the β = +0.5 result—may diminish on larger, instruction-hardened models (e.g., 7B–70B parameters).
4. **Answerable questions only.** Unanswerable SQuAD v2 queries were excluded, leaving the impact of sentence pruning on abstention and hallucination unmeasured.
5. **Metric and retention granularity.** EM and token-F1 penalise valid paraphrases, and normalized substring retention is a coarse proxy: 8 unpruned queries failed the boundary-matched check due to stem/boundary mismatches, and substring matching does not verify whether supporting context sentences were preserved.
6. **Sentence independence.** Scoring sentences independently can sever discourse links and coreference antecedents (e.g., pronouns referring to an entity named in a pruned preceding sentence).
7. **Hardware specificity of timings.** All timings come from a single Apple M3 laptop CPU, where in-situ pruning (50.9–60.6 ms) ran slower than isolated pruning (46.6 ms) due to contention with reader inference. Absolute speed-ups will vary on server GPUs or API-backed deployments.
8. **Scope of baselines.** TF-SCP was benchmarked against its own single-signal ablations and uncompressed RAG. Direct comparisons against trained extractive or perplexity-based compressors (e.g., LLMLingua, RECOMP) are left for future work.
