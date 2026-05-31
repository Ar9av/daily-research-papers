# AGENT-RADAR — Attention Steering for Multi-Agent Systems

> **Training-free plug-in that fixes "lost-in-the-middle" in multi-agent LLMs. +7.4 points average across 5 benchmarks.**

| | |
|---|---|
| **Paper** | AGENT-RADAR: Enhancing Multi-Agent Communication through Attention Steering with Context Relevance |
| **Authors** | Hongxiang Zhang, Yuan Tian, Tianyi Zhang |
| **Institution** | Purdue University |
| **arxiv** | [2605.30136](https://arxiv.org/abs/2605.30136) |
| **Date** | May 2026 |
| **Tags** | multi-agent, attention, context-management, training-free, LLM |

---

## The Problem

Multi-agent LLM systems accumulate long conversation histories as agents communicate. Relevant information gets buried — the "lost-in-the-middle" effect — causing hallucinations, logic drift, and degraded performance. This gets worse as you add agents or run more rounds. Existing fixes (compression, pruning) lose subtle but critical signals.

---

## The Idea

Don't compress or delete. Instead, **steer attention** toward the relevant parts of the history while keeping the full transcript intact. Score every sentence using three signals: semantic similarity to the current query, spatial decay (nearby agents in the communication graph matter more), and temporal decay (recent messages matter more). Inject the top sentences as attention anchors during inference.

```
History → score each sentence:
          semantic_sim × spatial_decay × temporal_decay
                      ↓
          top sentences → attention anchors
                      ↓
          Agent inference (full transcript still intact)
```

---

## Architecture

| Component | What it does |
|---|---|
| **Semantic Scorer** | Cosine similarity between sentence embedding and current query (all-MiniLM-L6-v2) |
| **Spatial Decay** | Exponential decay by hop distance in agent comm graph — direct neighbors = 1.0 |
| **Temporal Decay** | Exponential decay by message age — most recent = 1.0 |
| **Combined Score** | `spatial × temporal × semantic` — sentence must exceed threshold θ |
| **SPA (Selective Prompt Anchoring)** | Amplifies attention weights on selected sentences during generation |

---

## Results

| Benchmark | Vanilla MAS | Best Prior (AgentDropout) | AGENT-RADAR | Gain vs Vanilla |
|---|---|---|---|---|
| **MATH-500** | 80.60% | 82.60% | **88.80%** | +8.20 |
| **MMLU-Pro** | 60.20% | 67.40% | **69.40%** | +9.20 |
| **HotpotQA** | 75.07% | 73.14% | **80.78%** | +5.71 |
| **2WikiMultihop** | 71.14% | 78.45% | **80.81%** | +9.67 |
| **MuSiQue** | 35.47% | 37.06% | **39.72%** | +4.25 |
| **Average** | — | — | — | **+7.41** |

- GPTSwarm + AGENT-RADAR: **+12.87 F1** on MuSiQue
- AutoGen + AGENT-RADAR: **~+5 points** across benchmarks
- Robust across random, layered, and fully-connected agent topologies
- Performance **keeps improving** as agent count and rounds increase (unlike all baselines)

---

## Key Insight

The transformer attention mechanism is already a retrieval system — you don't need to delete context, just tell the model what to look at. The spatial + temporal decay factors encode two intuitions human collaborators apply naturally: trust nearby colleagues more, and weight recent information over stale.

---

## Builder Takeaway

If you're running debate loops, planning agents, or critic-refiner pipelines — your agents are already suffering from context dilution. AGENT-RADAR is plug-in and training-free: score the history between rounds, inject the top sentences as a high-attention prefix. Works with AutoGen, GPTSwarm, LangGraph, or any custom MAS.

---

## Scripts

| Script | What it shows |
|---|---|
| [`scripts/agent_radar.py`](scripts/agent_radar.py) | Spatial + temporal + semantic scoring, attention steering wrapper |
