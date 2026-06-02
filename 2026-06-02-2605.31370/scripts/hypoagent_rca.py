"""
HypoAgent: An Agentic Framework for Interactive Abductive Hypothesis Generation over Knowledge Graphs
arxiv: 2605.31370  —  Gao et al., HKUST, 2026

Install: pip install anthropic  (optional — demo runs with mock KG)
Run:     python scripts/hypoagent_rca.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


# ── Knowledge Graph stub ──────────────────────────────────────────────────────

@dataclass
class Triple:
    subject: str
    relation: str
    obj: str


@dataclass
class KnowledgeGraph:
    triples: list[Triple] = field(default_factory=list)

    def add(self, s: str, r: str, o: str) -> None:
        self.triples.append(Triple(s, r, o))

    def execute_fragment(self, relation: str, anchor: str | None = None) -> set[str]:
        """Return entities connected by `relation`, optionally filtered by anchor."""
        results: set[str] = set()
        for t in self.triples:
            if t.relation == relation:
                if anchor is None or t.obj == anchor:
                    results.add(t.subject)
        return results

    def neighbors(self, entities: set[str], depth: int = 1) -> dict[str, list[str]]:
        """Return candidate (relation, entity) pairs reachable from `entities`."""
        candidates: dict[str, list[str]] = {}
        frontier = set(entities)
        for _ in range(depth):
            next_frontier: set[str] = set()
            for t in self.triples:
                if t.subject in frontier or t.obj in frontier:
                    if t.relation not in candidates:
                        candidates[t.relation] = []
                    candidates[t.relation].append(t.obj if t.subject in frontier else t.subject)
                    next_frontier.add(t.obj)
            frontier = next_frontier
        return candidates


# ── Hypothesis and Fragment types ─────────────────────────────────────────────

@dataclass
class Fragment:
    relation: str
    anchor: str | None = None
    label: str = ""

    def __str__(self) -> str:
        if self.anchor:
            return f"{self.relation}(x, {self.anchor})"
        return f"{self.relation}(x)"


@dataclass
class Hypothesis:
    fragments: list[Fragment]

    def __str__(self) -> str:
        return " ∧ ".join(str(f) for f in self.fragments)

    def execute(self, kg: KnowledgeGraph) -> set[str]:
        """Conjunctive execution — intersection of all fragment answer sets."""
        if not self.fragments:
            return set()
        result = kg.execute_fragment(self.fragments[0].relation, self.fragments[0].anchor)
        for f in self.fragments[1:]:
            result &= kg.execute_fragment(f.relation, f.anchor)
        return result


# ── Metrics ───────────────────────────────────────────────────────────────────

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def overlap(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


# ── Intent Recognition Agent (simplified) ─────────────────────────────────────

@dataclass
class DialogueTurn:
    utterance: str
    conditions: list[str]
    hypothesis: Hypothesis | None
    score: float


class IntentRecognitionAgent:
    """
    Maps utterance + dialogue history → structured conditions.
    In production: call an LLM with the full dialogue context.
    Here: simple keyword extraction.
    """

    CONDITION_KEYWORDS: dict[str, str] = {
        "treat": "treats",
        "target": "targets",
        "mechanism": "has_mechanism",
        "disease": "associated_with_disease",
        "drug": "is_drug",
        "pathway": "involved_in_pathway",
    }

    def recognize(self, utterance: str, history: list[DialogueTurn]) -> list[str]:
        conditions: list[str] = []
        for keyword, relation in self.CONDITION_KEYWORDS.items():
            if keyword.lower() in utterance.lower():
                conditions.append(relation)

        # History-aware: if utterance says "explore more", carry forward previous relations
        if "explore more" in utterance.lower() or "more" in utterance.lower():
            if history:
                last = history[-1]
                conditions = last.conditions[:]
                print(f"  [IRA] Resolving 'explore more' from history: {conditions}")

        if not conditions:
            # Fallback: reuse last turn's conditions
            if history:
                conditions = history[-1].conditions[:]
            else:
                conditions = ["associated_with"]

        print(f"  [IRA] Utterance: '{utterance}' → conditions: {conditions}")
        return conditions


# ── Hypothesis Generation Agent ────────────────────────────────────────────────

class HypothesisGenerationAgent:
    """
    In production: a trained lightweight Transformer conditioned on (observed, conditions).
    Here: builds a hypothesis by combining one Fragment per condition.
    """

    def generate(self, observed: set[str], conditions: list[str], kg: KnowledgeGraph) -> Hypothesis:
        fragments = [Fragment(relation=rel, label=rel) for rel in conditions]
        hyp = Hypothesis(fragments=fragments)
        print(f"  [HGA] Generated: {hyp}")
        return hyp


# ── Root Cause Analysis Agent ─────────────────────────────────────────────────

@dataclass
class FragmentDiagnosis:
    fragment: Fragment
    answer_set: set[str]
    score: float
    supported: bool


class RootCauseAnalysisAgent:
    threshold: float = 0.5

    def diagnose_fragments(
        self, hypothesis: Hypothesis, observed: set[str], kg: KnowledgeGraph
    ) -> list[FragmentDiagnosis]:
        diagnoses: list[FragmentDiagnosis] = []
        for f in hypothesis.fragments:
            answer_set = kg.execute_fragment(f.relation, f.anchor)
            score = jaccard(answer_set, observed)
            supported = score >= self.threshold
            diagnoses.append(FragmentDiagnosis(f, answer_set, score, supported))
            status = "SUPPORTED" if supported else "UNSUPPORTED"
            print(f"  [RCAA:Fragment] {f} → score={score:.2f} [{status}]")
        return diagnoses

    def probe_neighborhood(
        self, observed: set[str], kg: KnowledgeGraph
    ) -> list[tuple[str, float]]:
        neighbors = kg.neighbors(observed, depth=2)
        candidates: list[tuple[str, float]] = []
        for relation, entities in neighbors.items():
            answer_set = kg.execute_fragment(relation)
            score = jaccard(answer_set, observed)
            candidates.append((relation, score))
            print(f"  [RCAA:Probe] candidate relation '{relation}' → score={score:.2f}")
        candidates.sort(key=lambda x: -x[1])
        return candidates

    def repair(
        self,
        hypothesis: Hypothesis,
        diagnoses: list[FragmentDiagnosis],
        candidates: list[tuple[str, float]],
    ) -> Hypothesis:
        repaired: list[Fragment] = []
        used_candidates: list[str] = [rel for rel, _ in candidates]

        for diag in diagnoses:
            if diag.supported:
                repaired.append(diag.fragment)
            else:
                # Replace with the top candidate not already in the hypothesis
                existing_relations = {f.relation for f in hypothesis.fragments}
                replacement = None
                for rel, score in candidates:
                    if rel not in existing_relations and rel not in [f.relation for f in repaired]:
                        replacement = Fragment(relation=rel, label=f"repaired:{rel}")
                        print(f"  [RCAA:Repair] {diag.fragment} → replaced with '{rel}' (score={score:.2f})")
                        break
                if replacement:
                    repaired.append(replacement)
                else:
                    # No replacement found — drop the fragment
                    print(f"  [RCAA:Repair] {diag.fragment} → dropped (no candidate)")

        return Hypothesis(fragments=repaired)


# ── HypoAgent orchestrator ────────────────────────────────────────────────────

class HypoAgent:
    def __init__(self, kg: KnowledgeGraph, threshold: float = 0.7, max_iter: int = 3):
        self.kg = kg
        self.threshold = threshold
        self.max_iter = max_iter
        self.ira = IntentRecognitionAgent()
        self.hga = HypothesisGenerationAgent()
        self.rcaa = RootCauseAnalysisAgent()
        self.rcaa.threshold = threshold * 0.6  # fragment threshold is looser

    def run(self, observed: set[str], utterance: str, history: list[DialogueTurn]) -> tuple[Hypothesis, float]:
        conditions = self.ira.recognize(utterance, history)
        hypothesis = self.hga.generate(observed, conditions, self.kg)

        for iteration in range(self.max_iter):
            answer_set = hypothesis.execute(self.kg)
            score = jaccard(answer_set, observed)
            print(f"\n[HypoAgent] Iteration {iteration+1}: score={score:.2f}, answer_set={answer_set}")

            if score >= self.threshold:
                print(f"[HypoAgent] Threshold reached — returning hypothesis.")
                return hypothesis, score

            print("[HypoAgent] Invoking Root Cause Analysis Agent...")
            diagnoses = self.rcaa.diagnose_fragments(hypothesis, observed, self.kg)
            candidates = self.rcaa.probe_neighborhood(observed, self.kg)
            hypothesis = self.rcaa.repair(hypothesis, diagnoses, candidates)

        final_score = jaccard(hypothesis.execute(self.kg), observed)
        return hypothesis, final_score


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Build a small biomedical-style knowledge graph
    kg = KnowledgeGraph()

    # Observed drugs: Infliximab, Adalimumab, Golimumab
    observed = {"Infliximab", "Adalimumab", "Golimumab"}

    # True relations (supported): treats RA and IBD
    for drug in observed:
        kg.add(drug, "treats", "RA")
        kg.add(drug, "treats", "IBD")

    # False relation (what the initial hypothesis gets wrong): targets IL-17
    kg.add("Secukinumab", "targets", "IL-17")
    kg.add("Ixekizumab", "targets", "IL-17")

    # Neighborhood evidence: other drugs treating IBD
    kg.add("Infliximab", "treats", "Crohns")
    kg.add("Adalimumab", "associated_with", "TNF")
    kg.add("Golimumab", "associated_with", "TNF")

    agent = HypoAgent(kg, threshold=0.7, max_iter=3)
    history: list[DialogueTurn] = []

    # Turn 1: user asks about treatment
    print("=" * 60)
    print("TURN 1: 'What do these drugs treat?'")
    print("=" * 60)
    hyp, score = agent.run(observed, "What do these drugs treat?", history)
    print(f"\nFinal hypothesis: {hyp}")
    print(f"Final Jaccard score: {score:.3f}")

    history.append(DialogueTurn(
        utterance="What do these drugs treat?",
        conditions=["treats"],
        hypothesis=hyp,
        score=score,
    ))

    # Turn 2: user asks to explore more
    print("\n" + "=" * 60)
    print("TURN 2: 'Explore more about the mechanism'")
    print("=" * 60)
    hyp2, score2 = agent.run(observed, "Explore more about the mechanism", history)
    print(f"\nFinal hypothesis: {hyp2}")
    print(f"Final Jaccard score: {score2:.3f}")
