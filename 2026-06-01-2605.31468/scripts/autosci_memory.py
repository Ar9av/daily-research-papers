"""
AutoSci: A Memory-Centric Agentic System for the Full Scientific Research Lifecycle
arxiv: 2605.31468  —  Qian et al., Peking University, 2026

Install: pip install anthropic  (optional — demo runs with mocks)
Run:     python scripts/autosci_memory.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


# ── Entity schemas ────────────────────────────────────────────────────────────

class LongTermType(str, Enum):
    TOPIC = "topic"
    PAPER = "paper"
    FOUNDATION = "foundation"
    CONCEPT = "concept"
    METHOD = "method"
    PEOPLE = "people"


class ActiveType(str, Enum):
    IDEA = "idea"
    EXPERIMENT = "experiment"
    MANUSCRIPT = "manuscript"
    REVIEW = "review"


IDEA_STATES = ["proposed", "testing", "tested", "validated", "failed"]
EXPERIMENT_STATES = ["planned", "running", "completed", "abandoned"]
MANUSCRIPT_STATES = ["drafting", "revised", "submitted", "final"]
REVIEW_STATES = ["received", "rebuttal_drafting", "revised", "decision"]

STATE_TRANSITIONS: dict[str, list[str]] = {
    "proposed": ["testing"],
    "testing": ["tested"],
    "tested": ["validated", "failed"],
    "validated": [],
    "failed": [],
    "planned": ["running"],
    "running": ["completed", "abandoned"],
    "completed": [],
    "abandoned": [],
    "drafting": ["revised"],
    "revised": ["submitted"],
    "submitted": ["final"],
    "final": [],
    "received": ["rebuttal_drafting"],
    "rebuttal_drafting": ["revised"],
    "decision": [],
}


@dataclass
class Entity:
    id: str
    etype: LongTermType | ActiveType
    content: dict[str, Any]
    state: str | None = None
    links: list[tuple[str, str]] = field(default_factory=list)  # (relation, target_id)

    def add_link(self, relation: str, target_id: str) -> None:
        self.links.append((relation, target_id))

    def transition(self, new_state: str) -> None:
        if self.state is None:
            raise ValueError(f"Entity {self.id} has no lifecycle state")
        allowed = STATE_TRANSITIONS.get(self.state, [])
        if new_state not in allowed:
            raise ValueError(
                f"Invalid transition {self.state} → {new_state} for {self.id}. "
                f"Allowed: {allowed}"
            )
        self.state = new_state


# ── Trust Guard ───────────────────────────────────────────────────────────────

TrustVerdict = Literal["PASS", "WARN", "BLOCK"]


@dataclass
class TrustGuard:
    """
    Validates every SciMem write.
    Form checks: schema fields, lifecycle states, link types, bidirectionality.
    Content checks: evidence support, consistency (requires LLM reviewer in prod).
    """

    REQUIRED_FIELDS: dict[LongTermType | ActiveType, list[str]] = field(
        default_factory=lambda: {
            LongTermType.PAPER: ["title", "authors", "abstract"],
            LongTermType.METHOD: ["name", "description"],
            LongTermType.CONCEPT: ["name", "definition"],
            ActiveType.IDEA: ["description", "novelty_check"],
            ActiveType.EXPERIMENT: ["hypothesis", "protocol"],
            ActiveType.MANUSCRIPT: ["title", "sections"],
        }
    )

    VALID_RELATIONS: set[str] = field(
        default_factory=lambda: {
            "introduces",
            "applies",
            "extends",
            "critiques",
            "grounds",
            "supports",
            "uses",
            "authored_by",
            "belongs_to",
        }
    )

    def validate_form(self, entity: Entity, graph: "SciMem") -> list[str]:
        issues: list[str] = []

        # Required fields
        required = self.REQUIRED_FIELDS.get(entity.etype, [])
        for f in required:
            if f not in entity.content:
                issues.append(f"missing required field: {f}")

        # Lifecycle state validity
        if entity.state is not None:
            all_states = (
                IDEA_STATES + EXPERIMENT_STATES + MANUSCRIPT_STATES + REVIEW_STATES
            )
            if entity.state not in all_states:
                issues.append(f"unknown lifecycle state: {entity.state}")

        # Link validity
        for relation, target_id in entity.links:
            if relation not in self.VALID_RELATIONS:
                issues.append(f"unknown relation type: {relation}")
            if target_id not in graph.entities:
                issues.append(f"dangling link to unknown entity: {target_id}")

        return issues

    def validate_content(self, entity: Entity, graph: "SciMem") -> list[str]:
        """
        In production: call an independent reviewer LLM here.
        Returns a list of content-level concerns.
        """
        concerns: list[str] = []
        # Minimal heuristic: ideas must have novelty_check populated
        if entity.etype == ActiveType.IDEA:
            nc = entity.content.get("novelty_check", "")
            if not nc or nc.strip() == "":
                concerns.append("idea has empty novelty_check — evidence missing")
        return concerns

    def validate(self, entity: Entity, graph: "SciMem") -> tuple[TrustVerdict, list[str]]:
        form_issues = self.validate_form(entity, graph)
        content_issues = self.validate_content(entity, graph)
        all_issues = form_issues + content_issues

        if form_issues:
            return "BLOCK", all_issues
        if content_issues:
            return "WARN", all_issues
        return "PASS", []


# ── SciMem ────────────────────────────────────────────────────────────────────

@dataclass
class SciMem:
    """
    Schema-governed research memory with two regions:
    - Long-Term Knowledge Memory: stable cross-project knowledge
    - Active Research Memory: fast-changing project artifacts
    """

    entities: dict[str, Entity] = field(default_factory=dict)
    quarantine: dict[str, tuple[Entity, list[str]]] = field(default_factory=dict)
    trust_guard: TrustGuard = field(default_factory=TrustGuard)

    def write(self, entity: Entity) -> TrustVerdict:
        verdict, issues = self.trust_guard.validate(entity, self)
        if verdict == "BLOCK":
            self.quarantine[entity.id] = (entity, issues)
            print(f"[TrustGuard] BLOCK  {entity.id}: {issues}")
        elif verdict == "WARN":
            self.entities[entity.id] = entity
            print(f"[TrustGuard] WARN   {entity.id}: {issues}")
        else:
            self.entities[entity.id] = entity
            print(f"[TrustGuard] PASS   {entity.id}")
        return verdict

    def add_bidirectional_link(
        self, from_id: str, relation: str, to_id: str, inverse_relation: str
    ) -> None:
        if from_id not in self.entities or to_id not in self.entities:
            raise KeyError(f"Both entities must exist before linking: {from_id}, {to_id}")
        self.entities[from_id].add_link(relation, to_id)
        self.entities[to_id].add_link(inverse_relation, from_id)

    def context_view(self, focal_id: str, hops: int = 2) -> dict[str, Entity]:
        """Return the subgraph within `hops` of `focal_id`."""
        visited: set[str] = set()
        frontier = {focal_id}
        for _ in range(hops):
            next_frontier: set[str] = set()
            for eid in frontier:
                if eid in self.entities:
                    visited.add(eid)
                    for _, target in self.entities[eid].links:
                        if target not in visited:
                            next_frontier.add(target)
            frontier = next_frontier
        visited.discard(focal_id)
        result = {focal_id: self.entities[focal_id]} if focal_id in self.entities else {}
        result.update({eid: self.entities[eid] for eid in visited if eid in self.entities})
        return result

    def consolidate(self) -> None:
        """
        SciEvolve /dream — compress stale entries, remove duplicates.
        Stub: in production, call an LLM to merge similar Concept/Method entities.
        """
        print(f"[SciEvolve /dream] Memory has {len(self.entities)} entities — consolidation would run here.")

    def to_summary(self) -> str:
        by_type: dict[str, int] = {}
        for e in self.entities.values():
            key = str(e.etype)
            by_type[key] = by_type.get(key, 0) + 1
        return json.dumps({"entities": by_type, "quarantined": len(self.quarantine)}, indent=2)


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mem = SciMem()

    # Add a topic
    topic = Entity(
        id="topic:gpu-kernels",
        etype=LongTermType.TOPIC,
        content={"name": "GPU Kernel Optimization", "scope": "Triton / CUDA operator efficiency"},
    )
    mem.write(topic)

    # Add a paper (valid)
    paper = Entity(
        id="paper:tritonbench-2024",
        etype=LongTermType.PAPER,
        content={
            "title": "TritonBench: Benchmarking GPU Kernels with Triton",
            "authors": ["Smith", "Jones"],
            "abstract": "We present TritonBench, a workload suite for Triton kernel evaluation.",
        },
    )
    mem.write(paper)

    # Add a method (valid)
    method = Entity(
        id="method:profiling-guided-optim",
        etype=LongTermType.METHOD,
        content={
            "name": "Profiling-Guided Iterative Optimization",
            "description": "Use ncu/nsys profiler output as structured JSON feedback to a code agent.",
        },
    )
    mem.write(method)

    # Link paper → method (bidirectional)
    mem.add_bidirectional_link("paper:tritonbench-2024", "applies", "method:profiling-guided-optim", "supports")

    # Add an idea with lifecycle state
    idea = Entity(
        id="idea:claude-code-profiling",
        etype=ActiveType.IDEA,
        content={
            "description": "Use Claude Code with profiling feedback for iterative Triton kernel refinement",
            "novelty_check": "Semantic Scholar: no prior work uses nsys JSON as structured agent feedback for Triton",
        },
        state="proposed",
    )
    mem.write(idea)

    # Attempt a bad idea (missing novelty_check → WARN, not BLOCK since form is valid)
    bad_idea = Entity(
        id="idea:missing-novelty",
        etype=ActiveType.IDEA,
        content={
            "description": "Another kernel idea with no novelty check",
            "novelty_check": "",  # empty
        },
        state="proposed",
    )
    mem.write(bad_idea)

    # Attempt an entity with a bad schema field → BLOCK
    malformed = Entity(
        id="paper:no-abstract",
        etype=LongTermType.PAPER,
        content={"title": "Missing Abstract Paper", "authors": ["A"]},  # missing abstract
    )
    mem.write(malformed)

    # Transition idea lifecycle
    idea.transition("testing")
    print(f"\nIdea state after transition: {idea.state}")

    # Get a context view for a skill
    view = mem.context_view("idea:claude-code-profiling", hops=2)
    print(f"\nContext view for skill (focal=idea, 2 hops): {list(view.keys())}")

    # Memory summary
    print(f"\nSciMem summary:\n{mem.to_summary()}")

    # Trigger /dream consolidation
    mem.consolidate()
