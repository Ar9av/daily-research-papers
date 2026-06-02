---
name: hypoagent-rca
description: Root Cause Analysis loop for structured agent outputs — decompose a failed artifact into fragments, validate each independently, probe for evidence, repair only the broken parts instead of regenerating from scratch.
trigger: When an agent generates structured outputs (hypotheses, code, SQL, plans) that can be validated piece-by-piece, and you want targeted repair rather than blind retry on failure
---

## When to use

- Agents that generate structured outputs: logic formulas, code, SQL queries, API plans, workflow steps
- Any pipeline where outputs can be decomposed into independently executable sub-components
- Multi-turn interactive refinement where the user's intent evolves across turns
- When retry loops are expensive (LLM calls, code execution, API calls) and targeted repair would save budget
- Knowledge graph traversal, biomedical hypothesis generation, program synthesis

## Pattern

1. **Generate** — produce the structured output (hypothesis, code, plan) from a conditioned generator
2. **Evaluate** — execute/validate the full output; compute a similarity or correctness score
3. **If passing** — return to user; done
4. **Fragment Decomposition** — break the output into independently executable sub-components
5. **Fragment Diagnosis** — execute each fragment against the validator; score each independently; classify as SUPPORTED or UNSUPPORTED
6. **Neighborhood Probing** — search the local context (KG neighborhood, adjacent API docs, related code paths) for candidate replacements for unsupported fragments
7. **Targeted Repair** — replace only the UNSUPPORTED fragments with high-scoring alternatives from probing; preserve SUPPORTED fragments intact
8. **Feed back** — pass repaired output + diagnosis into next generation round; loop until score passes or max iterations reached

## Implementation

```python
def rca_loop(
    generator,       # fn(conditions) -> structured_output
    evaluator,       # fn(output) -> float score
    fragmenter,      # fn(output) -> list[Fragment]
    fragment_eval,   # fn(fragment) -> float score
    neighbor_probe,  # fn(observed) -> list[Candidate]
    repairer,        # fn(output, bad_fragments, candidates) -> repaired_output
    conditions,
    threshold=0.8,
    max_iter=3,
):
    output = generator(conditions)
    for _ in range(max_iter):
        score = evaluator(output)
        if score >= threshold:
            return output, score

        fragments = fragmenter(output)
        diagnosed = [(f, fragment_eval(f)) for f in fragments]
        bad = [f for f, s in diagnosed if s < threshold]
        good = [f for f, s in diagnosed if s >= threshold]

        candidates = neighbor_probe(conditions.observed)
        output = repairer(output, bad, good, candidates, conditions)

    return output, evaluator(output)
```

See `scripts/hypoagent_rca.py` for a full runnable implementation with mock KG.

## Tuning

| Parameter | Default | Notes |
|-----------|---------|-------|
| Fragment score threshold | 0.8 (Jaccard) | Lower for exploratory tasks; raise for precision-critical outputs |
| Max repair iterations | 3 | Each iteration costs ~2–3 LLM calls for RCA; keep low |
| Neighborhood probe depth | 2 hops | More hops = richer candidates but quadratic KG expansion |
| Condition history window | Full dialogue | Trim for long conversations if context budget is tight |

## Pitfalls

| Old Approach | This Paper |
|-------------|------------|
| Generate → evaluate → retry from scratch | Generate → evaluate → diagnose fragment → repair only bad parts |
| Treat failure as binary (pass/fail) | Decompose failure: which fragment failed, why, what's nearby in the KG |
| Static single-turn conditions | History-aware IRA: resolves "explore more about this" from prior turns |
| Re-prompt with "try again, be better" | Synthesize targeted conditions from fragment diagnosis + neighborhood evidence |
| All fragments treated equally | Fragment importance ranked by individual execution score |

## Key numbers

- Single-turn Jaccard: 63.3 → 82.4 on PharmKG8k vs CtrlHGen baseline
- Single-turn Jaccard: 71.8 → 90.4 on BioKG
- Unconditional DBpedia50: 78.1 → 94.0 (no user conditions — self-induced repair)
- Multi-turn BioKG: 72.4 (w/o RCA) → 93.6 (w/ RCA)
- Ablation: fragment diagnosis contributes more than neighborhood probing when removed

## Source

- arXiv: https://arxiv.org/abs/2605.31370
- Code: https://github.com/HKUST-KnowComp/HypoAgent
