"""
Code-on-Graph: Iterative Programmatic Reasoning via LLMs on Knowledge Graphs
arxiv: 2606.03705  —  Ding et al., Chinese Academy of Sciences, 2026

Install: pip install anthropic  (optional — demo runs with mock KG and mock LLM)
Run:     python scripts/cog_schema_executor.py
"""

from __future__ import annotations

import re
import textwrap
import traceback
from dataclasses import dataclass, field
from typing import Any


# ── Knowledge Graph stub ──────────────────────────────────────────────────────

@dataclass
class Triple:
    head: str
    head_type: str
    relation: str
    tail: str
    tail_type: str


@dataclass
class KnowledgeGraph:
    triples: list[Triple] = field(default_factory=list)

    def add(self, head: str, head_type: str, relation: str, tail: str, tail_type: str) -> None:
        self.triples.append(Triple(head, head_type, relation, tail, tail_type))

    def retrieve_subgraph(self, topic_entities: list[str], depth: int = 2, top_k: int = 8) -> list[Triple]:
        visited: set[str] = set(topic_entities)
        result: list[Triple] = []
        frontier = set(topic_entities)
        for _ in range(depth):
            next_frontier: set[str] = set()
            for t in self.triples:
                if t.head in frontier or t.tail in frontier:
                    result.append(t)
                    next_frontier.add(t.head)
                    next_frontier.add(t.tail)
            frontier = next_frontier - visited
            visited |= frontier
        seen = set()
        deduped = []
        for t in result:
            key = (t.head, t.relation, t.tail)
            if key not in seen:
                seen.add(key)
                deduped.append(t)
        return deduped[:top_k * depth]


# ── Schema-to-Class Mapper ────────────────────────────────────────────────────

def _safe_var(name: str) -> str:
    """Turn any string into a valid Python identifier."""
    s = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if s and s[0].isdigit():
        s = "e_" + s
    return s or "unknown"


def schema_to_classes(subgraph: list[Triple]) -> str:
    """Convert a retrieved subgraph into Python class definitions."""
    type_relations: dict[str, dict[str, str]] = {}
    for t in subgraph:
        type_relations.setdefault(t.head_type, {})[t.relation] = t.tail_type
        type_relations.setdefault(t.tail_type, {})  # ensure tail types also get a class

    lines = ["# Auto-generated schema classes from KG subgraph", ""]
    for entity_type, relations in sorted(type_relations.items()):
        lines.append(f"class {entity_type}:")
        rel_params = ", ".join(f"{rel}: list = None" for rel in sorted(relations))
        sep = ", " if rel_params else ""
        lines.append(f"    def __init__(self, name: str{sep}{rel_params}):")
        lines.append(f"        self.name = name")
        for rel in sorted(relations):
            lines.append(f"        self.{rel} = {rel} or []")
        lines.append(f"    def __repr__(self): return f'{entity_type}({{self.name}})'")
        lines.append("")
    return "\n".join(lines)


def build_instantiation_code(subgraph: list[Triple]) -> str:
    """Generate Python that creates class instances from retrieved triples."""
    entity_relations: dict[str, dict[str, list[str]]] = {}
    entity_types: dict[str, str] = {}
    for t in subgraph:
        entity_relations.setdefault(t.head, {}).setdefault(t.relation, []).append(t.tail)
        entity_types[t.head] = t.head_type
        entity_types.setdefault(t.tail, t.tail_type)

    lines = ["entities = {}"]
    for entity, etype in sorted(entity_types.items()):
        relations = entity_relations.get(entity, {})
        rel_args = ", ".join(
            f"{rel}={vals!r}" for rel, vals in sorted(relations.items())
        )
        var = _safe_var(entity)
        sep = ", " if rel_args else ""
        lines.append(f"{var} = {etype}(name={entity!r}{sep}{rel_args})")
        lines.append(f"entities[{entity!r}] = {var}")
    lines.append("")
    return "\n".join(lines)


# ── Sandbox executor with self-correction ────────────────────────────────────

def run_in_sandbox(full_code: str) -> Any:
    namespace: dict = {}
    exec(compile(full_code, "<cog_sandbox>", "exec"), namespace)
    return namespace.get("result")


def execute_with_correction(
    operation_code: str,
    class_defs: str,
    instantiation_code: str,
    llm_fix_fn,
    max_retries: int = 3,
) -> tuple[Any, bool]:
    code = operation_code
    for attempt in range(max_retries):
        full_code = class_defs + "\n" + instantiation_code + "\n" + code
        try:
            result = run_in_sandbox(full_code)
            if result is not None and result != [] and result != {}:
                return result, True
            print(f"  [Sandbox] Attempt {attempt+1}: empty result")
        except Exception:
            error_trace = traceback.format_exc()
            print(f"  [Sandbox] Attempt {attempt+1} error: {error_trace.splitlines()[-1]}")
            code = llm_fix_fn(code, class_defs, error_trace)
    return None, False


# ── Mock LLM ──────────────────────────────────────────────────────────────────

class MockLLM:
    def generate_code(self, question: str, subtask: str, class_defs: str) -> str:
        if "second biggest" in question.lower():
            return textwrap.dedent("""
                states = [e for e in entities.values() if hasattr(e, 'area') and e.area]
                sorted_states = sorted(states, key=lambda s: float(s.area[0]) if s.area else 0, reverse=True)
                result = [sorted_states[1].name] if len(sorted_states) > 1 else []
            """).strip()
        if "renegade" in question.lower() or "fight song" in question.lower():
            return textwrap.dedent("""
                result = []
                for entity in entities.values():
                    if hasattr(entity, 'fight_song') and 'Renegade' in entity.fight_song:
                        result.extend(entity.championships)
            """).strip()
        return "result = list(entities.keys())[:3]"

    def fix_code(self, broken_code: str, class_defs: str, error_trace: str) -> str:
        return broken_code  # mock: no-op fix

    def should_stop(self, result: Any) -> bool:
        return result is not None and result != []


# ── CoG orchestrator ──────────────────────────────────────────────────────────

class CodeOnGraph:
    def __init__(self, kg: KnowledgeGraph, llm: MockLLM, max_steps: int = 3, max_retries: int = 3):
        self.kg = kg
        self.llm = llm
        self.max_steps = max_steps
        self.max_retries = max_retries

    def run(self, question: str, topic_entities: list[str]) -> Any:
        print(f"\n[CoG] Question: {question}")
        history: list[tuple[str, Any]] = []

        for step in range(self.max_steps):
            print(f"\n[CoG] Step {step+1}")
            subgraph = self.kg.retrieve_subgraph(topic_entities)
            print(f"  [Retriever] {len(subgraph)} triples retrieved")

            class_defs = schema_to_classes(subgraph)
            instantiation = build_instantiation_code(subgraph)
            op_code = self.llm.generate_code(question, f"step {step+1}", class_defs)
            print(f"  [CodeGen] {op_code.splitlines()[0][:60]}...")

            result, ok = execute_with_correction(
                op_code, class_defs, instantiation, self.llm.fix_code, self.max_retries
            )
            history.append((f"step {step+1}", result))
            print(f"  [Executor] result={result} ok={ok}")

            if ok and self.llm.should_stop(result):
                print(f"\n[CoG] Answer: {result}")
                return result

        last = history[-1][1] if history else None
        print(f"\n[CoG] Answer (max steps): {last}")
        return last


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Demo 1: "second biggest state" — requires offset/ranking, breaks fixed argmax operators
    kg1 = KnowledgeGraph()
    for state, area in [("California", "423970"), ("Texas", "695662"),
                        ("Alaska", "1723337"), ("Montana", "380831")]:
        kg1.add(state, "State", "area", area, "Number")
        kg1.add(state, "State", "located_in", "United States", "Country")

    cog1 = CodeOnGraph(kg1, MockLLM())
    cog1.run("What is the second biggest state in the United States?", ["United States"])

    print("\n" + "=" * 60)

    # Demo 2: "Super Bowl won by team with fight song Renegade"
    kg2 = KnowledgeGraph()
    kg2.add("Pittsburgh Steelers", "SportsTeam", "fight_song", "Renegade", "Song")
    kg2.add("Pittsburgh Steelers", "SportsTeam", "championships", "Super Bowl XIII", "Championship")
    kg2.add("Pittsburgh Steelers", "SportsTeam", "championships", "Super Bowl XLIII", "Championship")
    kg2.add("Pittsburgh Steelers", "SportsTeam", "championships", "Super Bowl XIV", "Championship")
    kg2.add("Renegade", "Song", "performed_by", "Styx", "Artist")

    cog2 = CodeOnGraph(kg2, MockLLM())
    cog2.run("What was the last Super Bowl won by the team with the fight song Renegade?",
             ["Pittsburgh Steelers", "Renegade"])

    print("\n" + "=" * 60)
    print("Example schema-to-class output for KG2:")
    print(schema_to_classes(kg2.retrieve_subgraph(["Pittsburgh Steelers"])))
