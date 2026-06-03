---
name: cog-schema-executor
description: Schema-to-class mapping for LLM reasoning over structured data — convert entity/relation schemas to Python class definitions, generate executable code against the interface, inject real data as objects at runtime.
trigger: When an LLM needs to reason over structured external data (KGs, databases, APIs) without injecting raw records into the prompt; when predefined tool operators lack the expressiveness for complex queries; when context limits are a bottleneck for multi-hop reasoning
---

## When to use

- LLM + knowledge graph question answering with multi-hop or complex aggregation
- Any agent that retrieves structured records (SQL rows, API JSON, graph triples) and needs to reason over them
- When fixed tool/operator inventories can't express the required logic (ranking with offsets, nested filtering, custom aggregation)
- When retrieved data volume would exceed the prompt context budget
- Iterative reasoning tasks where each step builds on prior results non-linearly

## Pattern

1. **Retrieve subgraph** — BFS from topic entities; use semantic similarity (DistilBERT or equivalent) to rank and prune edges; depth=2, top-K per hop
2. **Map schema to classes** — for each predicate, infer head/tail entity types from schema constraints; create a Python class per entity type; relations become typed list attributes on the domain class
3. **Generate code** — prompt the LLM with: question + subtask + class definitions (no raw data); LLM writes Python that traverses, filters, sorts, aggregates over class instances
4. **Execute in sandbox** — instantiate classes with real triples; run generated code; capture result or exception traceback
5. **Self-correct** — on failure, feed traceback back to LLM for code regeneration; retry up to N times
6. **Evaluate** — binary stop/continue: has the current result plausibly answered the question? If yes, return; if no, go back to step 1 with a new subtask
7. **Non-linear chaining** — each subtask carries a parent-step reference; generated code can access results from any prior step, not just the immediately previous one

## Implementation

```python
def schema_to_classes(triples: list[tuple]) -> str:
    """Convert KG triples into Python class definitions."""
    # Group by entity type (head_type, relation, tail_type)
    # Each entity type → class; each relation → list attribute
    # Returns a Python source string with class definitions

def execute_with_correction(
    code: str,
    class_defs: str,
    instantiation_code: str,
    llm,
    max_retries: int = 3,
) -> tuple[any, bool]:
    """Run code in sandbox; on error, ask LLM to fix and retry."""
    for attempt in range(max_retries):
        try:
            full_code = class_defs + "\n" + instantiation_code + "\n" + code
            result = run_in_sandbox(full_code)
            if result is not None and result != []:
                return result, True
        except Exception as e:
            code = llm.fix_code(code, class_defs, str(e))
    return None, False

def cog_loop(question, kg, llm, max_steps=5):
    history = []
    for step in range(max_steps):
        subtask = llm.next_subtask(question, history)
        subgraph = retrieve_subgraph(kg, subtask.entities, subtask.predicates)
        class_defs = schema_to_classes(subgraph)
        instantiation = build_instantiation_code(subgraph)
        code = llm.generate_code(question, subtask, class_defs, history)
        result, ok = execute_with_correction(code, class_defs, instantiation, llm)
        history.append((subtask, result))
        if llm.should_stop(question, history):
            break
    return llm.summarize_answer(question, history)
```

See `scripts/cog_schema_executor.py` for a full runnable implementation.

## Tuning

| Parameter | Default | Notes |
|-----------|---------|-------|
| Subgraph depth | 2 | Covers CVT relations in Freebase; increase for denser KGs |
| Subgraph breadth (top-K edges) | 8 | Tradeoff between coverage and schema noise |
| Self-correction retries | 3 | More retries help complex queries; diminishing returns beyond 5 |
| Planning temperature | 1.2 | Higher for exploration; subtask diversity matters |
| Code generation temperature | 0.3 | Lower for determinism; correctness > variety here |
| Evaluation temperature | 0.0 | Must be deterministic for consistent stop decisions |

## Pitfalls

| Old Approach | CoG |
|-------------|-----|
| Inject raw triples into prompt | Map schema to Python classes; inject data at runtime |
| Fixed operator inventory (argmax, filter, hop) | LLM writes task-specific code; no operator limits |
| Linear subtask chain | Non-linear: each subtask has a parent-step reference, can build on any prior result |
| Retry on failure with same prompt | Feed error traceback → targeted code correction |
| All retrieved facts in context | Only class definitions in context; facts as in-memory objects |

## Key numbers

- WebQSP: 88.7 vs 83.9 (PoG) — +4.8pp
- CWQ: 79.1 vs 72.6 (PoG) — +6.5pp
- GrailQA Overall: 93.5 vs 78.8 (SRP) — +14.7pp
- GrailQA Zero-shot: 84.2 vs 78.3 (PoG) — +5.9pp
- Max improvement over prior SOTA: +10.5pp

## Source

- arXiv: https://arxiv.org/abs/2606.03705
