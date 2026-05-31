# MUSE-Autoskill — Self-Evolving Agents via Skill Lifecycle

> **Agents that accumulate reusable skills over time — and automatically retire the bad ones.**

| | |
|---|---|
| **Paper** | MUSE-Autoskill: Self-Evolving Agents via Skill Creation, Memory, Management, and Evaluation |
| **Authors** | Huawei Lin, Peng Li, Jie Song, Fuxin Jiang, Tieying Zhang |
| **arxiv** | [2605.27366](https://arxiv.org/abs/2605.27366) |
| **Date** | May 2026 |
| **Tags** | agents, skill-learning, memory, self-improvement, multi-agent |

---

## The Problem

LLM agents today rediscover the same solutions every run. There's no accumulation — a skill learned on task 100 isn't available on task 101 unless you hardcode it. Existing skill libraries are static, manually curated, and don't self-clean. Skills that degrade (because the environment changed, or were just wrong) stay in the library forever.

---

## The Idea

Give agents a **lifecycle for skills**: create → store → evaluate → update or retire. After solving any task, the agent extracts a reusable skill. A background evaluator scores stored skills on new tasks. Skills below a quality threshold get retired automatically. Skills above it get updated with new examples.

```
Task → Solve → Extract Skill → Store in Library
                                      ↓
                              Evaluate on new tasks
                                      ↓
                          Score > threshold? Update : Retire
```

---

## Architecture

| Component | What it does |
|---|---|
| **Skill Creator** | After solving a task, extracts a generalized, reusable skill description |
| **Skill Memory** | Vector store of skills — retrieved by embedding similarity to new task |
| **Skill Evaluator** | Runs stored skills on held-out tasks, scores success rate |
| **Skill Updater** | Rewrites skill descriptions with new successful examples |
| **Skill Retiree** | Removes skills that fall below a quality threshold |

---

## Results

- Consistent improvement over no-skill baseline across all evaluated task domains
- Self-evolving library outperforms static manually-curated skill sets
- Retirement loop measurably reduces noise in retrieval as library grows
- Works across coding, reasoning, and tool-use tasks

---

## Key Insight

The evaluation + retirement loop is what makes this work. Without it, skill libraries become noisy over time — too many mediocre skills dilute the good ones. MUSE treats skills like code: they need tests (evaluator), they need maintenance (updater), and bad ones should be deleted (retiree).

---

## Builder Takeaway

If you're building agents that run repeatedly across similar tasks — coding agent, research agent, customer-support agent — add a skill layer. After each successful task, extract a skill. After N tasks, run the evaluator loop. The 100th task benefits from everything learned on tasks 1–99.

---

## Scripts

| Script | What it shows |
|---|---|
| [`scripts/skill_lifecycle.py`](scripts/skill_lifecycle.py) | Skill create → evaluate → update/retire loop |
