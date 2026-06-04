---
title: "Your Agents Are Aging Too: Agent Lifespan Engineering for Deployed Systems"
arxiv_id: "2605.26302"
date: "2026-06-04"
authors: "Jianing Zhu, Yeonju Ro, John T. Robertson, Kevin Wang, Junbo Li, Haris Vikalo, Aditya Akella, Zhangyang \"Atlas\" Wang"
institution: "The University of Texas at Austin"
tags: ["agent-memory", "long-lived-agents", "benchmark", "reliability", "memory-systems", "evaluation"]
---

# Your Agents Are Aging Too: Agent Lifespan Engineering for Deployed Systems

> **Frozen weights don't mean frozen behavior. A deployed agent's effective state keeps drifting as it compresses, retrieves, revises, and gets maintained — and reliability decays over its lifespan in four distinct, separately-repairable ways.**

| Field | Value |
|-------|-------|
| Authors | Jianing Zhu, Yeonju Ro, et al. (8 authors) |
| Institution | The University of Texas at Austin |
| arXiv | [2605.26302](https://arxiv.org/abs/2605.26302) |
| Project | [AgingBench.github.io](https://agingbench.github.io/) |
| Date | May 25, 2026 |
| Tags | agent-memory, long-lived-agents, reliability, evaluation |

---

## The Problem

We evaluate agents like freshly-booted models — one snapshot, day one, done. But deployed agents run for weeks and months. A coding agent carries repo context across hundreds of tasks. An enterprise assistant tracks decisions over a quarter. A personal planner accumulates preferences, budgets, contacts.

The trap: even with frozen model weights, the agent's *effective state* keeps changing. It compresses old history, retrieves from a growing store, revises facts, and undergoes routine maintenance. So reliability is a property of the **whole harness over time**, not a snapshot of the base model. Day-one benchmarks completely miss the question that matters in production: how long does this thing stay reliable, and when it breaks, *what* broke?

---

## The Idea

AgingBench reframes long-lived reliability as **Agent Lifespan Engineering (ALE)** — three questions:

1. How long does a deployed agent stay reliable?
2. *How* does it decay — compression, interference, revision, or maintenance?
3. *Where* should you repair — write, retrieval, utilization, or the memory lifecycle?

The core taxonomy splits aging into four mechanisms across two families:

```
ACCUMULATION-DRIVEN (gets worse as state grows)
  ├─ Compression  →  write-time summary drops a future-relevant detail   →  OMISSION
  └─ Interference →  similar memories crowd out the target fact           →  CONFUSION

EVENT-DRIVEN (triggered by discrete changes)
  ├─ Revision     →  changed/derived state not updated correctly          →  STALENESS
  └─ Maintenance  →  flush / recompaction / prompt swap breaks behavior    →  COLLAPSE
```

Two machines make this measurable:

**Temporal dependency DAG** — generators emit a graph `G = (Facts, Edges, Interference pairs)`: version chains (facts supersede each other), dependency edges (probes depend on facts many sessions apart), confusable entity pairs, and lifecycle events at controlled times. Seed-reproducible sweeps over session count, update rate, chain depth, interference density.

**Counterfactual probe ladder** — to locate the failing stage, replace pipeline components with oracles:

```
P1  agent write + agent retrieval + agent utilize   →  Acc_P1  (baseline)
P2  agent write + ORACLE retrieval + agent utilize   →  Acc_P2
P3  ORACLE context (gold facts in prompt)            →  Acc_P3

Read error  (interference) = Acc_P2 − Acc_P1   ← oracle retrieval recovers it
Write error (compression)  = Acc_P3 − Acc_P2   ← survives oracle retrieval
Util error  (revision)     = 1 − Acc_P3        ← gold facts in-context, still wrong
```

Maintenance error is observationally aliased with write error (both = missing facts) so it's separated *temporally*: measure the write-error jump immediately across a lifecycle event.

---

## Architecture

| Component | What It Does |
|-----------|--------------|
| Temporal FactGraph (`G`) | Version chains, dependency edges (chain-depth `d`), interference pairs, accumulators `Σ`, lifecycle events `e_k` |
| Programmatic generators | Emit `G` + task stream; seed-reproducible sweeps over session count, density, update rate |
| Session loop | Each session `t`: read `M_t` → answer task `τ_t` + held-out probes `q_t` → compress `M_{t+1} = U(M_t, H_t; θ)` |
| Aging curve | Per-session scores `m(t)`; yields half-life `t½`, decay slope (OLS), hazard proxy |
| Memory pipeline | `History →W→ Store →R→ Context →U→ Answer` — W=write/compress, S=store, R=retrieve, U=utilize |
| Counterfactual ladder | P1/P2/P3 oracle swaps → stage-level diagnostic profile (Read/Write/Util shares) |
| Temporally-aware scoring | Each metric tied to a DAG structure → mechanism-specific curves, not one recall number |

---

## Results

| Finding | Number |
|---------|--------|
| Max keyword-recall drop over 10 sessions (GPT-4o-mini, frozen weights) | **85%** |
| Half-life spread from memory policy alone (same model) | **4.5×** |
| Performance cliff from a single maintenance event | **67%** |
| Total scale | ~400 runs, 8–200 sessions, 7 scenarios, 14 models |
| Aggregate error rates across S1/S2/S5 models | clustered tight: **0.60–0.82** |
| S1 (Research Lit) bottleneck | Utilization-dominated |
| S2 (Lifestyle) bottleneck | Write-dominated (value-preserving compaction needed) |
| S5 (Self-Plan) on gpt4o-mini vs llama | near-pure Write failure → large Read/Interference component |
| S2 nearly solved by Qwen | 0.21 error |
| Careful vs lossy compression half-life (Qwen3-8B, S1) | 5.9 → 7.4 sessions |

---

## Key Insight

**The same aggregate failure rate hides completely different root causes.** Three models can all sit at ~0.7 total error on a scenario, but one is failing at write-time (dropped the detail), one at retrieval (buried under similar entries), one at utilization (had the fact in-context and still answered wrong). The single "memory score" prescribes the same fix everywhere — "give it more memory" — when the actual repairs are unrelated: a value-preserving compaction prompt, a better retriever, or a planning-loop change that forces re-reads. And memory *policy* moves the half-life 4.5× — more than swapping the model does.

---

## Builder Takeaway

If you ship a long-lived agent, your day-one eval is lying to you about month-two reliability. Three concrete moves:

1. **Eval over a lifespan, not a snapshot.** Run the agent across many sessions with facts that supersede, accumulate, and get probed long after they're written. Track the *curve* (half-life), not a single accuracy number.
2. **Diagnose the stage before you fix.** When the agent is wrong, run the oracle ladder: oracle-retrieval and gold-context probes tell you whether to fix write-time compression, retrieval, or utilization. Don't reflexively add more memory.
3. **Treat maintenance as a release event.** Recompaction, history flush, and prompt swaps are silent regressions waiting to happen — gate them with before/after regression probes, like you'd gate a deploy.

The mental model: your agent is a system that accrues technical debt and stale indices over its lifetime. Engineer its lifespan, don't just benchmark its birth.

---

## Scripts

| Script | What It Demonstrates |
|--------|----------------------|
| [scripts/aging_curve.py](scripts/aging_curve.py) | Run an agent across N sessions with a compaction policy; compute the aging curve, half-life, and decay slope |
| [scripts/counterfactual_probe.py](scripts/counterfactual_probe.py) | The P1/P2/P3 oracle ladder — decompose a failure rate into Read/Write/Utilization error shares |
| [scripts/temporal_factgraph.py](scripts/temporal_factgraph.py) | Generate a temporal dependency DAG: version chains, interference pairs, accumulators, lifecycle events |
