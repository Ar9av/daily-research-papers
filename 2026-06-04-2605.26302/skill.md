---
name: agent-lifespan-diagnosis
description: Evaluate and diagnose reliability decay in long-lived agents over their deployment lifespan. Run an agent across many sessions, compute aging curves, and use an oracle counterfactual ladder to localize whether failures originate at write-time compression, retrieval/interference, utilization/revision, or maintenance events — so repairs target the actual failing stage instead of reflexively adding memory.
trigger: When building or operating an agent that persists state across many sessions (coding agents, enterprise assistants, personal planners); when an agent "was reliable but got worse over time"; when you need to decide what to fix in a memory pipeline; when evaluating memory policies or before/after a maintenance event (recompaction, flush, prompt swap).
---

## When to use

- Your agent runs for weeks/months and accumulates memory across sessions
- Reliability is fine on day-one evals but you suspect (or see) degradation in production
- You're choosing or tuning a memory/compaction policy and want to measure its effect on longevity
- An agent gives a wrong answer and you don't know whether to fix the writer, retriever, or reasoner
- You're about to run a lifecycle event (recompaction, history flush, model rotation, prompt update) and want a regression gate
- You need a metric better than a single end-to-end "memory score"

## Pattern

1. **Build a temporal dependency graph** — generate facts with version chains (facts supersede), dependency edges (probes depend on facts from sessions ago), interference pairs (confusable entities), and accumulators (derived state like `budget = init + Σdeltas`). Inject lifecycle events at controlled session times.
2. **Run the session loop** — at each session `t`: agent reads compressed memory `M_t`, answers the task + held-out probes, then compresses history into `M_{t+1} = U(M_t, H_t; θ)`. Score each session.
3. **Compute the aging curve** — collect per-session scores `m(t)`. Derive half-life `t½` (sessions to 50% capability loss), decay slope (OLS fit), hazard proxy. The *curve* is the signal, not the mean.
4. **Run the oracle ladder on failures** — for held-out probes, evaluate three conditions:
   - `P1`: agent's own write + retrieval + utilization (baseline)
   - `P2`: oracle retrieval over the agent-written store (removes retrieval failures)
   - `P3`: gold facts injected into prompt (removes write + retrieval failures)
5. **Read the diagnostic profile** —
   - Read/Interference error = `Acc_P2 − Acc_P1` (oracle retrieval recovers it)
   - Write/Compression error = `Acc_P3 − Acc_P2` (survives oracle retrieval → info was lost at write)
   - Utilization/Revision error = `1 − Acc_P3` (gold facts in-context, still wrong)
6. **Isolate maintenance separately** — it aliases with write error, so measure it *temporally*: `ΔS = WriteError(t+) − WriteError(t−)` across the nearest pre/post-event probes.
7. **Target the repair to the dominant stage** — write-dominated → value-preserving compaction prompt; read-dominated → better retriever/dedup; util-dominated → force re-reads / explicit derived-state update.

## Implementation

```python
def aging_curve(agent, sessions, probes) -> dict:
    """Run N sessions, return scores + half-life + decay slope."""
    # see scripts/aging_curve.py

def diagnose(agent, oracle_retriever, gold_context, probes) -> dict:
    """P1/P2/P3 ladder → {read, write, util} error shares."""
    acc_p1 = eval_probes(agent, probes)
    acc_p2 = eval_probes(agent.with_retriever(oracle_retriever), probes)
    acc_p3 = eval_probes(agent.with_context(gold_context), probes)
    return {
        "read_err":  acc_p2 - acc_p1,
        "write_err": acc_p3 - acc_p2,
        "util_err":  1.0   - acc_p3,
    }
    # see scripts/counterfactual_probe.py
```

## Tuning

| Parameter | Meaning | Effect |
|-----------|---------|--------|
| `compaction_word_budget` | Words kept per compaction | Lower → more compression aging (omission) |
| `session_count` | Lifespan horizon | Longer → more accumulation aging surfaces |
| `interference_density` | # confusable entities per fact | Higher → more read/retrieval error |
| `update_rate` | How often facts get revised | Higher → more revision/utilization error |
| `chain_depth d` | Probe dependency depth | Deeper → harder relational recall |
| `lifecycle event time k` | When flush/recompact fires | Controls maintenance-shock measurement |

## Pitfalls

| Old approach | This paper |
|--------------|------------|
| Evaluate the agent once, day one | Evaluate across the full lifespan; track the aging curve |
| Report a single end-to-end memory score | Decompose into Read / Write / Utilization error shares |
| "Agent is wrong → give it more memory" | Diagnose the stage; the right fix is often *not* more memory |
| Assume frozen weights = frozen behavior | Effective state drifts via compression, retrieval, revision, maintenance |
| Treat maintenance (recompaction/flush) as safe | Treat it as a release event with before/after regression probes |
| Keyword recall catches all failures | Derived-state (accumulator) errors are invisible to keyword recall |

## Key numbers

- 85% max keyword-recall drop over 10 sessions with frozen weights (GPT-4o-mini)
- 4.5× half-life spread from memory policy alone — bigger lever than the model
- 67% performance cliff from a single maintenance event
- ~400 runs, 8–200 sessions, 7 scenarios, 14 models, runner-controlled + autonomous
- Aggregate error rates cluster at 0.60–0.82 while stage composition diverges wildly

## Source

- Paper: [arXiv 2605.26302](https://arxiv.org/abs/2605.26302)
- Project: [AgingBench.github.io](https://agingbench.github.io/)
