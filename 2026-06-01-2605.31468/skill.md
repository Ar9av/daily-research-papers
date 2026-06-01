---
name: autosci-memory
description: Build a persistent research memory with typed entities, lifecycle states, and Trust Guard validation — so long-horizon agents accumulate knowledge across sessions instead of starting from scratch.
trigger: When an agent needs persistent structured memory across sessions/projects, when managing long-horizon research or multi-step workflows, when agent outputs need to be validated before being stored
---

## When to use

- Multi-session research or analysis agents that should remember what they found last time
- Agents that accumulate scientific/domain knowledge and should not reconstruct context from scratch per run
- Systems where bad memory writes could corrupt downstream tasks (finance, legal, medical, code generation)
- Any agentic workflow with distinct lifecycle phases (research, experiments, writing, review) that need handoff artifacts

## Pattern

1. **Define typed entities** — don't store raw text; store schema-governed objects (e.g. Paper, Concept, Method, Experiment, Idea)
2. **Separate long-term from active memory** — stable knowledge goes in one region; fast-changing project artifacts go in another
3. **Model lifecycle states** — every active artifact has explicit states (proposed → testing → validated/failed); this makes the system resumable without reading chat history
4. **Gate all writes via Trust Guard** — schema validation + independent reviewer agent; PASS / WARN / BLOCK; quarantine blocked items
5. **Build typed relations** — not just entity pages but bidirectional links (Paper applies Method, Concept grounds Idea); enables graph traversal, not just keyword search
6. **Equip skills with tailored context views** — each downstream skill gets only the relevant memory subgraph, not the full graph
7. **Run evolution on the memory itself** — periodically compress stale entries, consolidate redundant material, promote validated findings to long-term

## Implementation

```python
# Entity types (Long-Term Knowledge Memory)
class EntityType(Enum):
    TOPIC = "topic"
    PAPER = "paper"
    FOUNDATION = "foundation"
    CONCEPT = "concept"
    METHOD = "method"
    PEOPLE = "people"

# Entity types (Active Research Memory)
class ActiveEntityType(Enum):
    IDEA = "idea"       # proposed → testing → validated/failed
    EXPERIMENT = "experiment"  # planned → running → completed/abandoned
    MANUSCRIPT = "manuscript"  # drafting → revised → submitted → final
    REVIEW = "review"   # received → rebuttal → revised → decision

# Trust Guard validates every write
class TrustGuard:
    def validate(self, entity, memory_graph) -> Literal["PASS", "WARN", "BLOCK"]:
        # Form checks: schema fields, lifecycle state, link types, bidirectionality
        # Content checks: evidence support, consistency with existing entities
        ...

# SciMem write path
def write_entity(entity, memory_graph, trust_guard):
    result = trust_guard.validate(entity, memory_graph)
    if result == "BLOCK":
        quarantine(entity)
    elif result == "WARN":
        flag_for_review(entity)
        memory_graph.add(entity)
    else:
        memory_graph.add(entity)
```

See `scripts/autosci_memory.py` for a full runnable implementation.

## Tuning

| Parameter | Default | Notes |
|-----------|---------|-------|
| Trust Guard strictness | WARN threshold | Tighten for medical/legal; loosen for rapid prototyping |
| Memory consolidation frequency | Per-project cycle | /dream should run after each major milestone |
| Context view depth | 2 hops from focal entity | More hops = richer context but larger token budget |
| Active entity retention | Until terminal state | Don't promote to long-term until validated/completed |

## Pitfalls

| Old Approach | This Paper |
|-------------|------------|
| Store memory as flat logs or vector chunks | Typed entity graph with schema-governed structure |
| One region for everything | Separate long-term (stable) from active (fast-changing) |
| Write anything, filter on read | Trust Guard blocks bad writes before they enter the graph |
| Memory dies with the session | Persistent across full project cycles; reused in future runs |
| Skills share a monolithic context | Each skill receives a tailored SciMem view |
| Feedback goes nowhere | SciEvolve converts signals into versioned system updates |

## Key numbers

- 10 typed entity types across 2 memory regions
- 20+ typed relations enabling graph traversal
- 30+ skills spanning 5 lifecycle stages
- 9 reusable DAG operators for multi-agent augmentation
- 6.3/10 automated ICLR score on autonomously generated GPU optimization paper
- 1.52× kernel speedup with feedback-driven iteration (1.58× on high-headroom tasks)
- 27 hours end-to-end (literature ingestion to manuscript artifact)

## Source

- arXiv: https://arxiv.org/abs/2605.31468
- Code: https://github.com/skyllwt/AutoSci
