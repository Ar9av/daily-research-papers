---
title: "MUSE-Autoskill: Self-Evolving Agents via Skill Creation, Memory, Management, and Evaluation"
arxiv_id: "2605.27366"
date: 2026-05-27
authors: ["Huawei Lin", "Peng Li", "Jie Song", "Fuxin Jiang", "Tieying Zhang"]
institution: "ByteDance Inc. + Rochester Institute of Technology"
tags: [agents, skills, memory, self-improvement, llm]
---

# MUSE-Autoskill

## Problem

LLM agents rely on reusable skills to solve complex tasks, but existing approaches treat skills as static, isolated artifacts — created once and never improved. There is no per-skill memory that accumulates experience across tasks, no unit-test validation before storing a skill, and flat conversation histories that overflow on long-horizon tasks. The result: skills that can't compound, can't be reliably reused, and get worse over time through drift.

## Solution

MUSE-Autoskill (Memory-Utilizing Skill Evolution) treats skills as long-lived, lifecycle-managed assets rather than one-off outputs. Skills are created on-demand from within the agent's runtime loop, validated through unit tests before storage, and continuously refined when tests fail. A novel skill-level memory layer accumulates per-skill experience across every task — wins, failures, adaptations — making future invocations progressively more reliable.

## Architecture

| Component | What it does |
|-----------|-------------|
| **skill_create tool** | Invoked from inside the runtime loop — skills are generated with full access to execution context |
| **Short-term memory** | Working memory for the current task session |
| **Long-term memory** | Persistent knowledge that survives across sessions |
| **Skill-level memory** | Novel: tracks every invocation, success, and failure per individual skill |
| **Evaluation subsystem** | Runs unit tests before a skill is stored; triggers auto-refinement on failure |
| **Context manager** | Adaptive compression + cross-session state persistence; prevents context-window overflow |
| **Cross-agent transfer** | Generated skills are portable — can be injected into a different agent without modification |

## Results

- **68.40%** overall accuracy on SkillsBench (51 real-world tasks) — best among all tested agents
- **+15.21 pp** lift over the no-skills baseline for MUSE
- **87.94%** accuracy on the 35 tasks where self-generated skills succeed — exceeds the human-skill ceiling
- **+10.51 pp** when MUSE-generated skills are transferred to a different agent (Hermes), closing **79%** of the gap to human-authored skills
- Beats Codex (67.3%) and Hermes (61.2%) on overall accuracy
- Best-in-class in 3 of 4 SkillsBench super-domains (Science & Engineering, Data Analysis, Ops & Planning)

## Key insight

Self-generated skills beat human-written ones. When MUSE creates skills from its own successful trajectories, it hits 87.94% — higher than the human-skill ceiling. The agent doesn't need humans to author skills; it needs enough task attempts to distill the pattern itself. This is the compounding effect: every task run makes the next one cheaper.

## Builder takeaway

If you're building agents that repeat similar sub-tasks across sessions and not persisting skill-level memory, you're recreating the same solutions from scratch every time. The fix isn't just storing skills — it's storing *what happened when you used them*. Unit-test before you store. Track per-skill history. Let the library grow.
