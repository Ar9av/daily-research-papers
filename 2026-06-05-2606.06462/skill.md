---
name: benchmark-agent
description: Autonomous benchmark construction pipeline — translate a plain-text evaluation requirement into a discriminative, evaluation-ready dataset through a Planner (design→ground→allocate) + Executor (plan→transform→verify) loop.
trigger: When you need to generate evaluation data, create a benchmark for a new capability, refresh a saturating benchmark, or build domain-specific test sets without manual annotation
---

## When to use

- You need evaluation data for a capability that no existing benchmark covers
- An existing benchmark is saturating and losing discriminative power
- You want to evaluate agent behavior, RAG quality, code correctness, or domain reasoning at scale
- Annotation is too slow/expensive for the data volume or refresh cadence you need
- You want model-family consistency checks (do larger models score higher?)

## Pattern

1. **State the requirement** in plain language: what capability, what modality, what difficulty level
2. **Design Agent** — decompose requirement into independently testable subtasks; iterate Propose → Revise → Discard until subtask set is well-posed
3. **Grounding Agent** — for each subtask: search candidate datasets, score transformability (alignment × robustness × signal preservation), reject infeasible subtask–dataset pairs
4. **Feasibility gate** — every subtask must have ≥1 valid (dataset, transformation) grounding; else return to Design Agent
5. **Allocation Agent** — assign sample quotas; diagnose bottlenecks; adjust until global constraints satisfied
6. **Executor: Sample Planning** — specialize dataset-level transformation plan to each raw sample's state
7. **Executor: Tool Execution** — run LLM-based and deterministic tools (TTS, OCR, image ops, web search) interleaved with planning
8. **Executor: Verification** — check semantic validity + format compliance; discard or locally correct failures; replenish until quotas met

## Implementation

```python
def benchmark_agent(requirement: str, dataset_pool: list[Dataset], quota: int) -> list[BenchmarkItem]:
    # Phase 1: Planning
    subtasks = design_agent(requirement)           # Propose → Revise → Discard loop
    grounded = grounding_agent(subtasks, dataset_pool)  # transformability validation
    assert all_feasible(grounded), "revise subtasks"
    plan = allocation_agent(grounded, quota)       # closed-loop quota resolution

    # Phase 2: Execution
    items = []
    for subtask, dataset, transform_plan, q in plan:
        batch = []
        while len(batch) < q:
            sample = dataset.sample()
            action_seq = sample_planner(sample, transform_plan)  # specialize to this sample
            result = execute_tools(sample, action_seq)           # LLM + deterministic tools
            if verify(result, subtask):
                batch.append(result)
            # else: discard or localized correction
        items.extend(batch)
    return items

def verify(item, subtask) -> bool:
    # semantic validity: does item actually test the subtask?
    # format compliance: answer schema, question structure
    # grounding: is the question answerable from the provided context?
    ...
```

See `scripts/benchmark_agent.py` for a full runnable implementation.

## Tuning

| Parameter | Default | Notes |
|-----------|---------|-------|
| Backbone LLM | GPT-5.1 / Claude-Sonnet | Closed-source models score ~2-5pp higher; open-source viable |
| Transformability score threshold | Filter lowest-scoring plans | Tighten for signal-sensitive domains (medical, code) |
| Verification strictness | Semantic + format | Add domain-specific checks (e.g. answer uniqueness for MCQ) |
| Replenishment cap | Until quota met | Set a max-retry limit to avoid infinite loops on sparse datasets |
| Subtask granularity | Independent, testable dimensions | Finer subtasks → more controllable but harder to ground |

## Pitfalls

| Old Approach | This Paper |
|-------------|------------|
| Manual annotation (5–6 min/sample) | Automated pipeline (0.2–0.3 min/sample, 20× faster) |
| Direct LLM generation (prompt → benchmark) | Agentic workflow with grounding + verification gates |
| Fixed benchmark, released once | Continuously refreshable; update on demand |
| One benchmark per team-months effort | Reusable pipeline across domains and modalities |
| Saturation forces full rebuild | Incremental refresh by re-running Executor on new data |
| Well-formed ≠ tests the right thing | Transformability validation + TSD metric ensure signal fidelity |

## Key numbers

- Human acceptance rate: 96–98% across 5 benchmark types
- LLM-as-Judge overall: 72–79 (Benchmark Agent) vs 54–59 (direct LLM generation)
- 20× speedup over human annotation
- Benchmark Agent quality stable across 4 backbone LLMs (range: 73–79)
- Removing transformability checking drops UIA from 68.54 → 44.30 on Omni-Understanding
- 15 benchmarks produced spanning text, audio, image, and omni-modal tasks

## Source

- arXiv: https://arxiv.org/abs/2606.06462
