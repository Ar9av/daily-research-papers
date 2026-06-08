---
title: "OPENSKILL: Open-World Self-Evolution for LLM Agents"
arxiv_id: "2606.06741"
date: "2026-06-08"
authors: "Zhiling Yan, Dingjie Song, Hanrong Zhang, Wei Liang, Yuxuan Zhang, Yutong Dai, Lifang He, Philip S. Yu, Ran Xu, Xiang Li, Lichao Sun"
institution: "Lehigh University; University of Illinois Chicago; UBC; Vector Institute; Salesforce AI Research; MGH / Harvard"
tags: ["agent-skills", "self-evolution", "open-world", "skill-synthesis", "LLM-agents", "supervision-free"]
---

# OPENSKILL: Open-World Self-Evolution for LLM Agents

> **Bootstraps a full skill-learning loop from only a task prompt and the open web — no curated skills, no trajectories, no verifier — and lands within 1–3pp of human performance.**

| Field | Value |
|-------|-------|
| Authors | Zhiling Yan, Dingjie Song, et al. |
| Institution | Lehigh University, UIC, Salesforce AI Research, Harvard |
| arXiv | [2606.06741](https://arxiv.org/abs/2606.06741) |
| Code | https://github.com/OpenLAIR/OpenSkill |
| Date | June 4, 2026 |
| Tags | agent-skills, self-evolution, supervision-free, open-world |

---

## The Problem

Self-evolving agents require a learning loop — but every existing approach assumes you already have part of one: curated skills, successful trajectories, or a verifier to judge improvement. Real deployments give you none of these, just a task prompt. Building both the skills *and* the verification signal from scratch, without ever seeing the ground-truth tests, is the unsolved problem.

---

## The Idea

OpenSkill treats the open web as both a knowledge source and a supervision-free practice environment. Three stages, separated by a leakage barrier:

```
Task prompt + environment
        ↓
Stage 1: Open-World Knowledge Acquisition
  → retrieve docs, repos, papers, tutorials → knowledge doc ki
  → synthesize structured skill plan pi
  → retrieve verification anchors kv (known row counts,
    metric ranges, documented output formats, domain standards)
        ↓
Stage 2: Leakage-Free Skill Evolution  [leakage barrier: GT tests hidden]
  → verifier LLM synthesizes deterministic pytest suite from kv
  → agent generates initial skill, executes, checks against virtual tests
  → on failure: diagnose (bug vs knowledge gap) → targeted retrieval
  → iterate up to J=3 rounds
        ↓
Stage 3: Zero-Shot Target Evaluation
  → frozen skill deployed to any target agent
  → GT tests unlocked here only
```

Skills are explicit portable artifacts (not weights), so they deploy to any model without retraining.

---

## Architecture

| Component | What It Does |
|-----------|-------------|
| Open-World Retriever D | Queries docs, repos, web for task-relevant knowledge; filters out benchmark identifiers to prevent leakage |
| Verification Retriever Dv | Retrieves independently checkable anchors: dataset statistics, API expected outputs, domain standards |
| Skill Planner | Synthesizes structured plan pi from (task, environment, knowledge doc) |
| Virtual-Task Verifier g | Separate LLM session; emits deterministic pytest suite from kv; never sees GT tests |
| Skill Evolution Loop | Agent generates, executes, receives structured diagnostic; retrieves missing knowledge on gap; iterates ≤J rounds |
| Diagnostic Classifier | LLM decides bug vs. knowledge gap; triggers targeted retrieval on gap |
| Leakage Barrier | Filters all queries during Stages 1–2; GT tests enter pipeline only at Stage 3 |

---

## Results

SkillsBench (11 domains), two target agents:

| Method | Opus 4.6 (Overall) | GPT 5.2 (Overall) |
|--------|-------------------|-------------------|
| No Skill | 25.5% | 25.0% |
| Self-Gen | 23.9% | 32.2% |
| CoT | 23.9% | 33.3% |
| AutoSkill | 24.7% | 11.2% |
| Memento | 30.1% | 15.6% |
| **OpenSkill** | **43.6%** | **42.1%** |
| Human upper bound | 44.5% | 44.8% |

- OpenSkill vs. strongest baseline: **+8.9pp** (Opus), **+8.8pp** (GPT)
- Best or tied-best in 8/11 domains (Opus), 7/11 domains (GPT)
- Within 1–3pp of human upper bound

SocialMaze / ScienceWorld (best automated):

| Benchmark | Opus 4.6 | GPT 5.2 |
|-----------|----------|---------|
| SocialMaze | 82.7% (+0.9 over best baseline) | 70.7% (+0.9) |
| ScienceWorld | 90.0% (+1.3) | 85.3% (+2.2) |

Cross-model transfer (Opus 4.6 skills → other models):
- Haiku 4.5: +10.4pp over no-skill; +5.5pp over AutoSkill
- Qwen 3 Coder: +13.0pp; +5.4pp
- DeepSeek V3: +14.8pp

Virtual Verifier quality:
- Covers **88.9%** of GT test intents (120/135) — zero GT access
- Generates **3.4× more** test functions than human-authored GT suite
- Statistical alignment with GT outcomes: OR=2.97, p=0.035

---

## Key Insight

The virtual verifier — built with no access to the ground-truth test suite — covers 88.9% of what the humans wrote, and adds 15.3 more assertions per task that the humans didn't include. Documentation and domain standards encode nearly everything you'd need to test a task, if you extract the right verification anchors. The open world isn't just a knowledge source; it's a supervision signal.

---

## Builder Takeaway

If you're building coding or tool-use agents, stop treating documentation as background reading and start treating it as a test-generation source. OpenSkill's verification anchor extraction — pulling out known row counts, expected metric ranges, documented output formats — is the concrete technique. The leakage-barrier pattern (build skills and tests from docs; evaluate against GT only at the end) is directly applicable to any agent evaluation pipeline where you want to prevent overfitting to the benchmark. Skills as portable artifacts (not weights) is also worth adopting: the same skill file works across Haiku, Qwen, DeepSeek, and Mistral without rewriting.

---

## Scripts

| Script | Description |
|--------|-------------|
| [scripts/openskill.py](scripts/openskill.py) | OpenSkill pipeline: open-world acquisition, virtual verifier, leakage-free evolution loop |
