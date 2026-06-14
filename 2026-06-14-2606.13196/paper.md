---
title: "Under What Conditions Can a Machine Become Genuinely Creative? A Designics-Based Requirement Framework for Creative Machine and Human–AI Co-Living"
arxiv_id: "2606.13196"
date: "2026-06-14"
authors: "Yong Zeng"
institution: "Concordia University, Montreal"
tags: ["machine-creativity", "AI-ethics", "framework", "designics", "agentic-AI", "human-AI-collaboration"]
---

# Under What Conditions Can a Machine Become Genuinely Creative?

> **Generating novel outputs ≠ genuine creativity. This paper derives 10 structural requirements from the mechanics of how creativity actually works. Most current AI systems satisfy a few. The critical ones — conflict identification, consequence observation, rescoping — are unsatisfied.**

| Field | Value |
|-------|-------|
| Author | Yong Zeng |
| Institution | Concordia University, Montreal |
| arXiv | [2606.13196](https://arxiv.org/abs/2606.13196) |
| Date | June 11, 2026 |
| Tags | machine-creativity, AI-ethics, designics, framework, agentic-AI |

---

## The Problem

AI creativity is evaluated by outputs: novelty, usefulness, surprise. These criteria don't distinguish a genuinely creative system from a powerful generative one. A system can produce surprising outputs through statistical recombination without representing the environment in which those outputs matter, without identifying what was insufficient, and without observing what its actions changed.

---

## The Idea

Ground machine creativity in the structure of how meaningful change actually unfolds — "Designics," the science of meaning-bearing intentional change.

Genuine creativity is not output generation. It is participation in **recursive intervention dynamics**:

```
Situation
  → Constrained perception (bounded intake of situational information)
  → Scoping (bounding the actionable environment)
  → Conflict identification (what is insufficient, blocked, or unrealized)
  → Intervention (introduce an active change into the environment)
  → Consequence observation (what did the intervention change?)
  → Knowledge and environment update (integrate what was learned)
  → Rescoping (adjust boundary for next cycle)
  → Repeat
```

This cycle distinguishes creativity from optimization: optimization searches within a fixed space; creativity changes the conditions of the space.

---

## The 10 Requirements

Organized through 3 laws of Designics:

### Law of Perception — what the system can represent, include, and revise

| # | Requirement | What it means |
|---|-------------|---------------|
| R1 | Environment representation | Operate within a represented environment, not just transform inputs |
| R2 | Scoped perception | Bound attention to what's relevant; revise that scope |
| R5 | Consequence observation | Observe what the intervention actually changed |
| R7 | Rescoping | Revise the boundary of perception in response to outcomes |
| R9 | Value-based scoping | Include ethical, social, ecological, legal constraints in what is perceived |

### Law of Conflict — what calls for intervention

| # | Requirement | What it means |
|---|-------------|---------------|
| R3 | Conflict identification | Identify blockages, instabilities, and unrealized possibilities without being told |

### Law of Capability — what the system can transform and sustain

| # | Requirement | What it means |
|---|-------------|---------------|
| R4 | Intervention capability | Enact changes that transform the environment, not just produce artifacts |
| R6 | Knowledge and environment update | Learn from observed consequences |
| R8 | Local-to-global unfolding | Build emergent global structure from principled local decisions |
| R10 | Human–AI co-living | Support human agency, trustworthy knowledge, sustainable cooperation |

---

## Contemporary AI as Pressure Cases

| System type | Satisfies | Lacks |
|-------------|-----------|-------|
| Foundation models (LLMs) | R4 (generate), partial R1 | R3, R5, R7, R10 |
| Agentic tool loops | R4, partial R5/R6 (via feedback) | R3 (prompt-specified conflicts), R7, R9 |
| Scientific discovery (FunSearch, AlphaFold) | R4, R8 (local-to-global) | R3, R7, R9 |
| Autonomous mesh generation (FREEMESH) | R1–R8 (geometric scope) | R9, R10 |
| Human–machine teaming with workload reallocation | R5, R10 | R3 (human-specified), R7 |

None satisfies all 10 requirements. The most critical gaps across all paradigms: **R3** (conflict identification), **R7** (rescoping), and **R9** (value-based scoping).

---

## Key Insight

**Autonomy ≠ creativity.** A system can execute plans, call tools, and optimize performance without being creative in this sense. Genuine creativity requires changing the conditions of action — identifying what is wrong without being told, observing what changed, and revising the frame for the next cycle.

The corollary for ethics: if a genuinely creative machine intervenes in human environments (not just produces artifacts for review), then ethical constraints cannot be a post-generation filter. They must be part of what the system perceives, what it identifies as conflict, and which interventions it considers viable.

---

## Builder Takeaway

Use the 10 requirements as an evaluation checklist for any AI system you call "creative":

- Does it identify what's wrong, or only solve what humans specify?
- Does it observe consequences of its own actions, or only produce outputs?
- Does it revise its scope when it learns something, or iterate within the same frame?
- Are ethical constraints internal to what it perceives, or bolted on afterward?

The gap between current agentic systems and genuine creativity is mostly at R3 (conflict identification) and R7 (rescoping). Systems that add structured environment observation, anomaly detection, and scope-revision loops to existing agents move toward these requirements.

---

## Scripts

| Script | Description |
|--------|-------------|
| [scripts/creativity_checklist.py](scripts/creativity_checklist.py) | Interactive checklist to evaluate any AI system against the 10 Designics requirements |
