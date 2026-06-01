---
title: "AutoSci: A Memory-Centric Agentic System for the Full Scientific Research Lifecycle"
arxiv_id: "2605.31468"
date: "2026-06-01"
authors: "Weitong Qian*, Beicheng Xu*, Zhongao Xie*, Bowen Fan, et al."
institution: "Peking University (PKUDAIR)"
tags: ["scientific-agents", "memory", "multi-agent", "self-evolution", "research-automation"]
---

# AutoSci: A Memory-Centric Agentic System for the Full Scientific Research Lifecycle

> **A persistent research environment that executes, remembers, and evolves across projects — and generates ICLR-ready papers end-to-end in ~27 hours.**

| Field | Value |
|-------|-------|
| Authors | Weitong Qian*, Beicheng Xu*, Zhongao Xie* et al. (Peking University) |
| Institution | PKUDAIR, Peking University |
| arXiv | [2605.31468](https://arxiv.org/abs/2605.31468) |
| Code | https://github.com/skyllwt/AutoSci |
| Date | May 29, 2026 |
| Tags | scientific-agents, memory, DAG-orchestration, self-evolution |

---

## The Problem

Every scientific research agent today is amnesiac. It runs one project, produces some artifacts, and starts the next from scratch — no memory of reviewer complaints, failed experiments, or what the field has already tried. Existing systems also skip key stages: most don't handle rebuttal, and none modify their own skills based on what went wrong. The gap is a unified system that can execute, remember, and evolve across full research lifecycles.

---

## The Idea

AutoSci builds four interlocking modules around a single idea: **research memory is the substrate, not a side-effect**.

```
Literature → Ideation → Experiment → Writing → Rebuttal
     ↕             ↕          ↕          ↕          ↕
              SciMem (typed knowledge graph)
                    ↕
              SciEvolve (/dream /forge /morph)
```

- **SciMem** — two-region typed memory: Long-Term Knowledge (topics, papers, concepts, methods, people, foundations) + Active Research (ideas, experiments, manuscripts, reviews), all schema-governed with Trust Guard validation
- **SciFlow** — 5-stage harness (Literature → Ideation → Experiment → Writing → Rebuttal), 30+ skills, state persisted outside LLM context so projects are resumable
- **SciDAG** — adaptive DAG of 9 reusable multi-agent operators (generate, variation, debate, refine, review, etc.) that augment hard stages without changing artifact contracts
- **SciEvolve** — `/dream` (memory evolution), `/forge` (skill versioning), `/morph` (DAG template improvement) convert feedback signals into auditable system updates

---

## Architecture

| Component | What It Does |
|-----------|-------------|
| SciMem / Long-Term Knowledge | Typed entity graph (10 types, 20+ relations); survives across projects; semantically addressable |
| SciMem / Active Research | Per-project workspace with lifecycle-state tracking on Idea, Experiment, Manuscript, Review entities |
| Trust Guard | Schema + evidence validation on every SciMem write; PASS / WARN / BLOCK; blocked artifacts quarantined |
| SciFlow harness | State, context, verification, feedback routing, orchestration — makes long-horizon research interruptible |
| SciDAG operators | 9 operator types in adaptive DAGs; templates versioned and updated by SciEvolve |
| SciEvolve /dream | Compresses, consolidates, archives stale memory periodically |
| SciEvolve /forge | Treats SciFlow skills as versioned protocols; patches them from failure traces |
| SciEvolve /morph | Prunes/specializes SciDAG templates based on operator performance traces |

---

## Results

| Metric | Value |
|--------|-------|
| GPU kernel case — ICLR automated review score | 6.3 / 10 |
| Biomedical drug discovery — ICLR automated review score | 5.8 / 10 |
| GPU case — end-to-end runtime | 27.3 hours |
| Biomedical case — end-to-end runtime | 22.6 hours |
| Kernel speedup (geomean, all baselines) | 1.52× |
| Kernel speedup (excluding degenerate baselines) | 1.18× |
| Feedback bonus — high-headroom cohort | 1.58× |
| Feedback bonus — broad cohort | 1.22× |
| Executability at iteration 5 | 157 / 157 (100%) |
| Ideas screened → selected | 5 generated → 1 selected (novelty + pilot) |

---

## Key Insight

AutoSci's GPU paper scored 6.3/10 on automated ICLR review — "weak accept" territory — generated fully autonomously in 27 hours. The non-obvious part isn't the score, it's that the review itself produced **actionable weaknesses** (limited external validity, missing comparator runs) that were stored back into SciMem as reusable submission-stage feedback for future projects. The system learns from its own rejections.

---

## Builder Takeaway

The architecture lesson here isn't "add memory" — it's **Trust Guard**. Every write to SciMem is schema-checked and evidence-reviewed by an independent agent before it joins the traversable graph. Most agent memory systems let bad information in silently; AutoSci quarantines it. If you're building any long-horizon agent where memory compounds across sessions, you need a write gate — not just a read retriever.

---

## Scripts

| Script | Description |
|--------|-------------|
| [scripts/autosci_memory.py](scripts/autosci_memory.py) | SciMem architecture: typed entity graph with Trust Guard validation |
