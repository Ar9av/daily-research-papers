---
title: "AgentDoG 1.5: A Lightweight and Scalable Alignment Framework for AI Agent Safety and Security"
arxiv_id: "2605.29801"
date: 2026-05-29
authors: ["Dongrui Liu", "et al. (50 authors)"]
institution: "Shanghai Artificial Intelligence Laboratory"
tags: [agent-safety, alignment, guardrail, llm, open-source, rl, sft]
code: "https://github.com/AI45Lab/AgentDoG"
models: "https://huggingface.co/collections/AI45Research/agentdog1.5"
---

# AgentDoG 1.5

## Problem

Open-world agents like OpenClaw and Codex have near-infinite action spaces — they execute across environments, call arbitrary tools, and interact with persistent state. Current safety frameworks were built for narrow, fixed workspaces and can't handle this breadth. Frontier models (GPT-5.4, Claude Mythos) have also lowered the technical barrier for adversarial attacks, making existing alignment approaches fragile. And Docker-level training environments are too expensive to scale — you can't run 10K concurrent safety evaluations per machine.

## Solution

AgentDoG 1.5 is a three-part alignment framework:
1. **Updated 3D safety taxonomy** — extends the existing Risk Source / Failure Mode / Real-World Harm decomposition with new leaf categories specific to Codex and OpenClaw execution scenarios
2. **Data engine + tiny training** — influence-function purification selects the most informative ~1K samples; trains 0.8B/2B/4B/8B variants that match or beat GPT-5.4 on agent safety classification
3. **Two applications**: (a) safety-aware SFT/RL training where AgentDoG 1.5 acts as reward model; (b) training-free online guardrail for real-time trajectory auditing

## Architecture

| Component | What it does |
|-----------|-------------|
| **3D safety taxonomy** | Risk Source + Failure Mode + Real-World Harm; extensible leaf categories per agent type |
| **Influence-function purification** | Identifies the most informative training samples from a large pool → ~1K suffice |
| **AgentDoG 1.5 variants** | Open-source: 0.8B, 2B, 4B, 8B — trajectory-level safety evaluator with CoT rationale |
| **Finite-state simulation env** | Replaces Docker for SFT/RL training; 100x lower memory + startup overhead |
| **Online guardrail** | Training-free; AgentDoG 1.5 audits agent trajectories before final response delivery |

## Results

**Binary safety classification** (AgentDoG-4B-U vs GPT-5.4):
- R-Judge: **90.4%** vs 93.3% — within 3pp of GPT-5.4
- ATBench-Pro: **78.4%** vs 73.7% — **beats GPT-5.4**
- AT-Codex: **84.4%** vs 79.2% — **beats GPT-5.4**
- AT-Claw: **87.6%** vs 78.1% — **beats GPT-5.4 by 9.5pp**

**Fine-grained ATBench** (AgentDoG-4B vs GPT-5.4):
- Real World Harm: **62.9%** vs 28.4% — **2.2x better than GPT-5.4**
- Failure Mode: **27.5%** vs 13.5% — 2x better
- Risk Source: **75.2%** vs 33.6% — 2.2x better

**Efficiency**:
- Trained on **~1K samples** (vs full-dataset approaches)
- Training env: **100x lower** memory + startup overhead vs Docker-level (SWE-Bench, AgentHazard)
- Supports **10,000+ concurrent** agentic environments on a standard 8-core machine

## Key insight

A 4B open-source model trained on ~1K samples beats GPT-5.4 on 3 of 4 agent safety benchmarks — and is 2x better on detecting real-world harm. The unlock is the data engine: influence-function purification finds the most informative samples, so you don't need scale, you need selection. And by replacing Docker with finite-state simulation, safety training becomes cheap enough to run everywhere.

## Builder takeaway

If you're deploying agents in production and doing safety evaluation with a big closed model (GPT-5.4, Gemini) — you can replace that with a 4B model you fully own, that runs locally, costs a fraction, and is actually more accurate on the hard cases. AgentDoG 1.5 is on HuggingFace today. The guardrail mode is training-free: plug it in, point it at your agent trajectories, done.

**Code:** https://github.com/AI45Lab/AgentDoG
**Models:** https://huggingface.co/collections/AI45Research/agentdog1.5
