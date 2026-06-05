---
title: "Benchmark Everything Everywhere All at Once"
arxiv_id: "2606.06462"
date: "2026-06-05"
authors: "Shiyun Xiong, Dongming Wu, Peiwen Sun, Yuang Ai, Bokang Yang, Wencheng Han, Xiao-Hui Li, Xiangyu Yue"
institution: "MMLab, CUHK; Beihang University; Huawei Technologies; Shandong University"
tags: ["benchmark-generation", "multi-agent", "evaluation", "multimodal", "autonomous-agents"]
---

# Benchmark Everything Everywhere All at Once

> **An autonomous agent that builds evaluation benchmarks on demand — describe what you want to test, get a dataset in 0.2 minutes per sample with 97% human acceptance.**

| Field | Value |
|-------|-------|
| Authors | Shiyun Xiong, Dongming Wu, Peiwen Sun, Yuang Ai, et al. |
| Institution | MMLab CUHK, Beihang University, Huawei Technologies |
| arXiv | [2606.06462](https://arxiv.org/abs/2606.06462) |
| Code | (demo + code linked in paper) |
| Date | June 4, 2026 |
| Tags | benchmark-generation, multi-agent, evaluation, multimodal |

---

## The Problem

Benchmarks saturate fast — Qwen models crossed 80% on MMLU, GSM8K, and MATH within a few release cycles. Building a new benchmark to reveal the next capability gap takes months of human annotation, can't be reused, and has to be rebuilt from scratch for each domain. The gap between how fast models improve and how fast the field can measure them is widening.

---

## The Idea

Benchmark Agent treats benchmark construction as an agentic pipeline, not a human annotation task. A user states what they want to evaluate in plain language; the system produces a ready-to-run benchmark.

```
User requirement (plain text)
        ↓
Benchmark Planner
  ├── Design Agent: brainstorm + refine evaluation subtasks
  ├── Grounding Agent: validate subtask ↔ real dataset ↔ transformation feasibility
  └── Allocation Agent: resolve quotas under global constraints
        ↓
Benchmark Executor
  ├── Sample-level Planning: specialize transformation plan per sample
  ├── Tool Execution: OCR, TTS, image resize, web search, file processing, noise injection
  └── Verification + Replenishment: quality gate; re-sample if item fails
        ↓
Evaluation-ready benchmark (items + answers + metadata)
```

The design–grounding feedback loop ensures every subtask has at least one feasible real-data realization before committing to execution. If not, the plan is rejected and revised upstream.

---

## Architecture

| Component | What It Does |
|-----------|-------------|
| Design Agent | Converts informal user requirement into structured, testable subtasks via Propose / Revise / Discard tools |
| Grounding Agent | For each subtask: searches candidate datasets, validates transformability (alignment × robustness × signal preservation), rejects infeasible subtasks back to Design |
| Allocation Agent | Assigns sample quotas under global constraints; diagnoses bottlenecks; adjusts until feasible or fails back |
| Benchmark Executor | Orchestrates sample-level realization through interleaved LLM planning + deterministic tool execution |
| Tool Pool | OCR, ASR/TTS, image processing, file conversion, web search, text translation, noise injection |
| Verification | Checks semantic validity + format compliance; routes failures to localized correction or re-generation; tracks quota fulfillment |

---

## Results

| Metric | Value |
|--------|-------|
| Human acceptance rate (across 5 benchmarks) | 96–98% |
| LLM-as-Judge overall (Benchmark Agent) | 72–79 |
| LLM-as-Judge overall (direct LLM generation, best) | 54–59 |
| Annotation time — human | 5–6 min / sample |
| Annotation time — Benchmark Agent | 0.2–0.3 min / sample |
| Speedup over human annotation | ~20× |
| Benchmark Agent backbone stability (overall score range) | 73–79 across 4 different LLM backends |
| Ablation: removing TC+Scoring drop (Omni-Understanding UIA) | 68.54 → 44.30 |
| MLLM consistency trend (Multi-Perspective, 2B→27B) | 71.06 → 87.23 |
| MLLM consistency trend (Art-Reasoning, 2B→27B) | 40.96 → 56.38 |

---

## Key Insight

Direct LLM generation (prompting GPT-5.4 or Claude with the same user requirement) produces structurally sound questions but scores ~17pp lower on user-intent alignment and target-signal dependency. A capable model can write a well-formed question; it cannot reliably build something that actually tests what you asked for. The agentic workflow — particularly transformability validation and sample-level planning — is what bridges the gap between "well-formed" and "evaluates the right thing."

---

## Builder Takeaway

If your eval pipeline is a static benchmark from 2023, you're measuring your model against a frozen past. The infrastructure to generate, refresh, and customize evaluation data on demand now exists. The Benchmark Agent pattern — Planner (design → ground → allocate) + Executor (plan → transform → verify) — is directly applicable to any domain where you need discriminative test data: code quality, agent behavior, RAG fidelity, safety. Build your eval loop the same way you build your training loop: automated, versioned, continuously updated.

---

## Scripts

| Script | Description |
|--------|-------------|
| [scripts/benchmark_agent.py](scripts/benchmark_agent.py) | Benchmark Planner pipeline: subtask design, grounding validation, allocation with feasibility loop |
