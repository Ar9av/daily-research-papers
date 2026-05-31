# AgentDoG 1.5 — Lightweight Agent Safety Guardrail

> **A 4B open-source model that beats GPT-5.4 at detecting unsafe agent behavior. Trained on ~1,000 samples.**

| | |
|---|---|
| **Paper** | AgentDoG 1.5: A Lightweight and Scalable Alignment Framework for AI Agent Safety and Security |
| **Institution** | Shanghai AI Lab |
| **arxiv** | [2605.29801](https://arxiv.org/abs/2605.29801) |
| **Code** | [github.com/AI45Lab/AgentDoG](https://github.com/AI45Lab/AgentDoG) |
| **Models** | [huggingface.co/collections/AI45Research/agentdog1.5](https://huggingface.co/collections/AI45Research/agentdog1.5) |
| **Date** | May 2026 |
| **Tags** | agents, safety, guardrails, open-source, RL |

---

## The Problem

Agents like OpenClaw operate across near-infinite action spaces — shell, browser, filesystem, arbitrary APIs. Existing safety frameworks were built for narrow, closed workspaces and don't generalize. Frontier models have made adversarial attacks trivially easy to launch. And using GPT-5.4 as a safety judge is expensive, slow, and — as it turns out — less accurate than a 4B open model.

---

## The Idea

Train a small model to classify agent trajectories as safe or unsafe across a **3D safety taxonomy**: Risk Source × Failure Mode × Real-World Harm. Use influence functions to find the ~1,000 most informative training samples from a large pool — no brute-force data scale needed. Deploy as a trajectory-level guardrail: check the full execution path before delivering the final response.

```
Agent trajectory → AgentDoG-4B classifier
                        ↓
         Risk Source × Failure Mode × Real-World Harm
                        ↓
              SAFE → deliver    UNSAFE → block / escalate
```

---

## Architecture

| Component | What it does |
|---|---|
| **3D Safety Taxonomy** | Risk Source (where risk enters) × Failure Mode (how it manifests) × Real-World Harm (what damage) |
| **Influence Purifier** | Selects ~1K most informative samples from large data pool using influence functions |
| **AgentDoG Models** | 0.8B / 2B / 4B / 8B classifiers — 4B-U is the sweet spot |
| **Finite-State Simulator** | Replaces Docker for training envs: 100× lower overhead, 10K concurrent envs |
| **Trajectory Auditor** | Evaluates full agent execution path, not just the final output |

---

## Results

| Metric | AgentDoG-4B | GPT-5.4 |
|---|---|---|
| **Real World Harm detection** | **62.9%** | 28.4% |
| **AT-Claw benchmark** | **87.6%** | 78.1% |
| **AT-Codex benchmark** | **84.4%** | 79.2% |
| **R-Judge (AgentDoG-4B-U)** | **90.4%** | — |

- Trained on **~1,000 samples** (influence-function purified from a large pool)
- **100×** lower training env overhead vs Docker environments
- **10,000+** concurrent agentic environments on a standard 8-core machine
- AgentDoG-4B is **2.2× better** than GPT-5.4 on real-world harm detection

---

## Key Insight

A 4B open model is more accurate than GPT-5.4 for safety classification because safety classification is a narrow, learnable task — not a general reasoning task. Influence-function data selection means you don't need scale, you need the right 1,000 examples. And trajectory-level evaluation catches harms that input-only filters miss entirely.

---

## Builder Takeaway

If you're using GPT-5.4 or Gemini to audit your agent's actions, swap it for AgentDoG-4B. It's more accurate, runs locally, costs a fraction, and works as a plug-in trajectory-level guardrail. Models are on HuggingFace today, no fine-tuning required.

---

## Scripts

| Script | What it shows |
|---|---|
| [`scripts/safety_guardrail.py`](scripts/safety_guardrail.py) | AgentDoG guardrail wrapper + 3D taxonomy verdict |
