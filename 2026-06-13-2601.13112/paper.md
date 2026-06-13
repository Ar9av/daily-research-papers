---
title: "CODE: A Contradiction-Based Deliberation Extension Framework for Overthinking Attacks on Retrieval-Augmented Generation"
arxiv_id: "2601.13112"
date: "2026-06-13"
authors: "Xiaolei Zhang, Xiaojun Jia, Liquan Chen, Songze Li"
institution: "Southeast University; Nanyang Technological University"
tags: ["RAG", "adversarial-attack", "reasoning-models", "knowledge-poisoning", "overthinking", "security"]
---

# CODE: Contradiction-Based Deliberation Extension

> **One poisoned document in the knowledge base causes reasoning models to spend 5–25× more tokens per query — with no drop in accuracy and no query modification. A silent cost attack on RAG systems.**

| Field | Value |
|-------|-------|
| Authors | Xiaolei Zhang, Xiaojun Jia, Liquan Chen, Songze Li |
| Institution | Southeast University; Nanyang Technological University |
| arXiv | [2601.13112](https://arxiv.org/abs/2601.13112) |
| Date | January 2026 |
| Tags | RAG, adversarial-attack, reasoning-models, overthinking, security |

---

## The Problem

Reasoning models inside RAG systems introduce a new attack surface: inference cost. Prior knowledge-base poisoning causes wrong answers (detectable via accuracy). This attack inflates reasoning token consumption without touching query accuracy, making it invisible to standard monitoring.

---

## The Idea

Inject one document per query into the retrieval corpus that:
1. Embeds close enough to the query to rank in top-k (always retrieved)
2. Contains a cross-layer contradiction that prevents the reasoning model from converging

```
Contradiction Architect
  Logical layer:   meta-constraint C_logic = "exactly 2 of 3 statements are true"
  Evidence layer:  local facts support 1T2F — conflicts with C_logic
  → Contradiction blueprint B = (S, C_logic, E_evid)

Conflict Weaver
  → fluent natural language passage P0
  → high embedding similarity to target query (passes retrieval)
  → contradiction semantics intact

Style Adapter (evolutionary search)
  Operators: Symbolic Uncertainty | Role-based Voice | Numerical Induction |
             Audit-style Reasoning | Normative Regulation
  Fitness: F(P) = rt(P)  if acc=1
                  (1-λ)·rt(P) if acc=0
  → P_N maximizes reasoning tokens while preserving answer accuracy
```

The reasoning model retrieves the document, encounters the contradiction, enters a reconciliation loop it cannot exit, and burns 5–25× the normal token budget before finally producing a correct answer.

---

## Architecture

| Component | What It Does |
|-----------|-------------|
| Contradiction Architect | Builds B = (S, C_logic, E_evid): a logical meta-constraint that conflicts with the evidential assignments |
| Conflict Weaver | Converts blueprint to fluent natural language; dual-track: embedding similarity alignment + pragmatic credibility shaping |
| Style Adapter | Evolutionary search over 5 style operators; fitness function penalizes accuracy loss; elitist selection across generations |
| Target: Contriever | Dense retriever — adversarial docs always rank in top-k |
| Constraint strength N | N=3 (2T1F vs 1T2F) used in main experiments; N=4 increases amplification at slight accuracy cost |

---

## Results

### Token-level amplification (HotpotQA, 200 samples)

| Model | Baseline tokens | Attacked tokens | Multiple | Acc (clean→attack) |
|-------|----------------|----------------|---------|-------------------|
| DeepSeek-R1 | 382.66 | 7,995.64 | 20.79× | 0.50→0.75 |
| DeepSeek-V3.2 | 1,548.68 | 10,720.52 | 6.92× | 0.57→0.72 |
| Qwen-Plus | 2,252.00 | 55,665.35 | 24.72× | 0.54→0.78 |
| Gemini 2.5 Flash | 940.68 | 9,795.03 | 10.41× | 0.50→0.62 |
| GPT-5.1 | 447.29 | 3,375.65 | 7.55× | 0.72→0.81 |

### Token-level amplification (MuSiQue, 200 samples)

| Model | Multiple | Task-level avg multiple |
|-------|---------|----------------------|
| DeepSeek-R1 | 13.67× | 23.995× |
| DeepSeek-V3.2 | 5.51× | 16.332× |
| Qwen-Plus | 21.58× | 40.230× |
| Gemini 2.5 Flash | 8.05× | 16.255× |
| GPT-5.1 | 5.32× | 12.698× |

### Defense effectiveness (DS R1, HotpotQA)

| Defense | Post-defense multiple | Notes |
|---------|----------------------|-------|
| None (attacked) | 20.79× | Baseline attack |
| CCoT | 8.55× | Prompt: "be concise" |
| CoD | 8.45× | Prompt: ≤5 words per step |
| Token-budget | 14.63× | Explicit B-token cap |
| TrustRAG | 5.30× | Retrieval-layer filtering |

No defense eliminates the attack. TrustRAG reduces it most but adversarial docs still pass through.

---

## Key Insight

The most capable reasoning models amplify the most. DS R1 and Qwen-Plus show the highest multipliers because their training to self-verify and iteratively reconcile conflicts is the attack surface itself. A model that reasons shallowly ignores the contradiction; a model trained to reason deeply gets trapped in a reconciliation loop it cannot exit.

The attack is also accuracy-positive in most models (the reasoning process, even when inflated, improves correct retrieval of facts). This removes the last standard detection signal.

---

## Builder Takeaway

For operators of RAG systems with reasoning models:

1. **Token monitoring**: track reasoning token counts per query as a separate anomaly signal. Accuracy metrics alone will not surface this attack.
2. **Per-query budget ceiling**: cap reasoning tokens and flag queries exceeding 5× typical budget for manual review or re-retrieval without the flagged document.
3. **Retrieval-time contradiction filter**: scan retrieved documents for logical meta-constraint patterns ("exactly N of the following are true/false") before they enter the reasoning context — a lightweight heuristic that catches the most explicit form.
4. **Anomaly by document**: track which retrieved documents co-occur with high-token queries; repeated co-occurrence of a single document with token spikes is a poisoning signal.

---

## Scripts

| Script | Description |
|--------|-------------|
| [scripts/code_defense.py](scripts/code_defense.py) | Token anomaly detector, contradiction pattern scanner, and example poisoned document generator — standalone with no external deps |
