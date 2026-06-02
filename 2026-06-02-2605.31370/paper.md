---
title: "HypoAgent: An Agentic Framework for Interactive Abductive Hypothesis Generation over Knowledge Graphs"
arxiv_id: "2605.31370"
date: "2026-06-02"
authors: "Yisen Gao, Yixi Cai, Tianshi Zheng, Jiaxin Bai, Yangqiu Song"
institution: "HKUST, Beihang University, Hong Kong Baptist University"
tags: ["knowledge-graphs", "abductive-reasoning", "multi-agent", "hypothesis-generation", "root-cause-analysis"]
---

# HypoAgent: An Agentic Framework for Interactive Abductive Hypothesis Generation over Knowledge Graphs

> **Three agents collaborating to generate, refine, and surgically repair logical hypotheses over knowledge graphs — targeted fragment-level diagnosis beats blind retry.**

| Field | Value |
|-------|-------|
| Authors | Yisen Gao, Yixi Cai, Tianshi Zheng, Jiaxin Bai, Yangqiu Song |
| Institution | HKUST, Beihang University, HKBU |
| arXiv | [2605.31370](https://arxiv.org/abs/2605.31370) |
| Code | https://github.com/HKUST-KnowComp/HypoAgent |
| Date | May 29, 2026 |
| Tags | knowledge-graphs, abductive-reasoning, multi-agent, root-cause-analysis |

---

## The Problem

Existing controllable hypothesis generators over knowledge graphs treat failure as a binary signal: the output either matches or doesn't, and if not, you sample again. There's no diagnosis of which part of the hypothesis was wrong, no way to track evolving user intent across a conversation, and no surgical repair — just regeneration from scratch. Multi-turn interactions compound this: "explore more about this" has no meaning to a single-turn system.

---

## The Idea

HypoAgent puts three agents in sequence, with the Root Cause Analysis Agent activating only when needed:

```
User utterance + dialogue history
        ↓
Intent Recognition Agent (IRA)
  → structured KG conditions {relation, entity, pattern, ...}
        ↓
Hypothesis Generation Agent (HGA)
  → first-order logic hypothesis H over KG
  → execute H → answer set A(H)
  → if sim(A(H), observations) OK → return to user
  → else ↓
Root Cause Analysis Agent (RCAA)
  ├── Fragment Diagnosis: decompose H, execute each fragment, score individually
  ├── KG Neighborhood Probing: find candidate relations/entities near observed entities
  └── Hypothesis Refinement: repair weak fragments or synthesize new conditions
        ↓ (loop back to HGA)
```

The IRA is history-aware: it jointly considers the current utterance and previous hypotheses/conditions so "keep the entity but change the relation" resolves correctly. The RCAA distinguishes reliable fragments from faulty ones rather than treating the whole hypothesis as wrong.

---

## Architecture

| Component | What It Does |
|-----------|-------------|
| Intent Recognition Agent (IRA) | Maps natural-language utterance + dialogue history → structured conditions (relation, entity, relationnumber, entitynumber, pattern) |
| Hypothesis Generation Agent (HGA) | Invokes a lightweight 6-layer Transformer to generate first-order logic hypotheses conditioned on IRA output; maintains turn-level memory |
| Root Cause Analysis Agent (RCAA) | Decomposes failed hypotheses into executable fragments; probes KG neighborhood of observations; proposes targeted condition corrections |
| Fragment Diagnosis | Executes each hypothesis fragment separately against the KG; scores each against the observation set to isolate the broken part |
| KG Neighborhood Probing | Searches shared neighborhoods of observed entities for candidate relations/paths not present in the original hypothesis |
| Hypothesis Refinement | Repairs weak fragments with neighborhood evidence or generates refined conditions for the next generation round |

---

## Results

| Setting | Metric | Baseline (CtrlHGen / AbductiveKGR) | HypoAgent (best) |
|---------|--------|--------------------------------------|-----------------|
| Single-turn BioKG | Jaccard | 71.8 | 90.4 |
| Single-turn PharmKG8k | Jaccard | 63.3 | 82.4 |
| Single-turn DBpedia50 | Jaccard | 78.1 | 94.0 |
| Single-turn PharmKG8k | Overlap | 81.2 | 94.0 |
| Multi-turn BioKG w/ RCA | Jaccard | 72.4 (w/o RCA) | 93.6 |
| Multi-turn PharmKG8k w/ RCA | Jaccard | 81.5 (w/o RCA) | 94.2 |
| Multi-turn DBpedia50 w/ RCA | Jaccard | 75.3 (w/o RCA) | 90.7 |
| Unconditional DBpedia50 | Jaccard | 78.1 (CtrlHGen) | 94.0 |

Ablation (PharmKG8k): full RCA > fragment-diagnosis-only > neighborhood-probing-only > no RCA. Fragment diagnosis contributes more than neighborhood probing.

---

## Key Insight

In the unconditional setting — no user conditions provided — HypoAgent still substantially outperforms baselines. The system analyzes its own initial hypothesis, induces useful conditions from the failure, and self-improves without any external guidance. The performance gain isn't from richer user signals; it's from the agent's ability to decompose and diagnose its own output.

---

## Builder Takeaway

The RCA pattern here — decompose output into fragments, execute each independently, identify which ones fail, repair only those — is a general debugging loop for any agent that generates structured artifacts (code, SQL, logical plans, API call sequences). The instinct in most pipelines is "generate → evaluate → retry." HypoAgent shows "generate → evaluate → diagnose fragment → targeted repair" is strictly better. If you're building agents that produce structured outputs that can be validated piece-by-piece, this is the loop to implement.

---

## Scripts

| Script | Description |
|--------|-------------|
| [scripts/hypoagent_rca.py](scripts/hypoagent_rca.py) | Root Cause Analysis loop: fragment diagnosis + neighborhood probing + targeted repair |
