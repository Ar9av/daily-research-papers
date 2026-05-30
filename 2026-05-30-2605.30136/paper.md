---
title: "AGENT-RADAR: Enhancing Multi-Agent Communication through Attention Steering with Context Relevance"
arxiv_id: "2605.30136"
date: "2026-05-30"
authors: ["Hongxiang Zhang", "Yuan Tian", "Tianyi Zhang"]
institution: "Purdue University"
tags: [multi-agent, attention, context-management, training-free, LLM]
---

## Problem

Multi-agent LLM systems accumulate extremely long conversation histories as agents communicate. Relevant information gets buried in the middle of these histories — the "lost-in-the-middle" effect — causing hallucinations, logical errors, and performance degradation. This gets worse as you add more agents or run more rounds. Existing fixes either compress the history (losing subtle signals) or prune agents/edges (discarding useful context).

## Solution

AGENT-RADAR is a training-free plug-in that steers each agent's attention toward the most relevant parts of the conversation history without modifying or deleting anything. It scores every sentence in the transcript using three combined signals: semantic similarity to the current query, spatial decay (messages from nearby agents in the communication graph matter more), and temporal decay (recent messages matter more). The top sentences are passed as an explicit attention anchor during inference via Selective Prompt Anchoring (SPA).

## Architecture

| Component | Description |
|---|---|
| Spatial Decay | Exponential decay over graph hop distance — direct neighbors get full weight, distant agents get less |
| Temporal Decay | Exponential decay over message age — most recent message = 1.0, older messages discounted |
| Sentence Retrieval | Segments each prior message into sentences, encodes with all-MiniLM-L6-v2, scores by cosine sim to current query |
| Combined Score | `spatial_decay × temporal_decay × semantic_similarity` — sentence must score above threshold θ |
| Attention Steering | Selected sentences passed to SPA: amplifies attention weights on the chosen context without pruning the full transcript |

## Results

- MATH-500 (competition math): **88.80%** vs vanilla 80.60% (+8.20), vs AgentDropout 82.60% (+6.20)
- MMLU-Pro (general reasoning): **69.40%** vs vanilla 60.20% (+9.20), vs AgentDropout 67.40% (+2.00)
- HotpotQA (multi-hop QA): **80.78%** vs vanilla 75.07% (+5.71), vs AgentDropout 73.14% (+7.64)
- 2WikiMultihopQA: **80.81%** vs vanilla 71.14% (+9.67)
- MuSiQue: **39.72%** vs vanilla 35.47% (+4.25)
- Average gain over vanilla: **+7.41** absolute points across 5 benchmarks
- GPTSwarm + AGENT-RADAR: +12.87 F1 on MuSiQue
- AutoGen + AGENT-RADAR: ~+5 points on most benchmarks
- Works across random, layered, and fully-connected agent topologies
- Robust as agent count and round count increase (unlike baselines that degrade)

## Key insight

Compression and pruning lose information — but you don't need to delete anything. The transformer's attention mechanism is already a retrieval system; you can just *tell it what to look at* by boosting attention weights on the relevant sentences while keeping the full history intact. The spatial + temporal decay factors encode two things human collaborators naturally do: trust nearby colleagues more, and weight recent information over stale.

## Builder takeaway

If you're building multi-agent pipelines that run multiple rounds (debate, critique, planning loops), your agents are already suffering from context dilution. AGENT-RADAR is plug-in and training-free: add it between rounds by scoring the history with spatial/temporal/semantic signals, then inject the top sentences as a high-attention prefix. No retraining, no architecture change, works with AutoGen/GPTSwarm/any framework.
