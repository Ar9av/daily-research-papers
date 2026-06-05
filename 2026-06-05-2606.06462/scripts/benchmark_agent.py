"""
Benchmark Everything Everywhere All at Once (Benchmark Agent)
arxiv: 2606.06462  —  Xiong et al., MMLab CUHK, 2026

Install: pip install anthropic  (optional — demo runs with mocks)
Run:     python scripts/benchmark_agent.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Data types ────────────────────────────────────────────────────────────────

class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    OMNI = "omni"


@dataclass
class Subtask:
    id: str
    description: str
    modality: Modality
    evaluation_dimension: str
    groundings: list["Grounding"] = field(default_factory=list)
    quota: int = 0

    def is_grounded(self) -> bool:
        return len(self.groundings) > 0


@dataclass
class Dataset:
    name: str
    modality: Modality
    size: int
    domain: str
    samples: list[dict] = field(default_factory=list)

    def sample(self) -> dict:
        if self.samples:
            return random.choice(self.samples)
        return {"id": f"{self.name}_{random.randint(0, self.size)}", "content": f"Sample from {self.name}"}


@dataclass
class TransformationPlan:
    steps: list[str]
    alignment_score: float    # how well it matches the subtask intent (0-1)
    robustness_score: float   # how reliable the transformation is (0-1)
    signal_score: float       # how well it preserves the evaluation signal (0-1)

    @property
    def overall_score(self) -> float:
        return (self.alignment_score + self.robustness_score + self.signal_score) / 3

    def is_valid(self, threshold: float = 0.6) -> bool:
        return self.overall_score >= threshold


@dataclass
class Grounding:
    subtask_id: str
    dataset: Dataset
    plan: TransformationPlan


@dataclass
class BenchmarkItem:
    subtask_id: str
    question: str
    answer: str
    context: dict
    metadata: dict
    verified: bool = False


# ── Design Agent ──────────────────────────────────────────────────────────────

class DesignAgent:
    """
    Converts informal user requirement into structured, testable subtasks.
    In production: iterative LLM calls with Propose / Revise / Discard tools.
    """

    DIMENSION_TEMPLATES = [
        ("comprehension", "tests understanding of {domain} content"),
        ("reasoning", "tests multi-step reasoning over {domain} inputs"),
        ("grounding", "tests ability to ground answers in provided {modality} context"),
        ("discrimination", "tests fine-grained distinction between {domain} concepts"),
    ]

    def propose(self, requirement: str, modalities: list[Modality]) -> list[Subtask]:
        subtasks = []
        for i, (dim, desc_tmpl) in enumerate(self.DIMENSION_TEMPLATES[:3]):
            modality = modalities[i % len(modalities)]
            subtasks.append(Subtask(
                id=f"subtask_{i}",
                description=desc_tmpl.format(domain="general", modality=modality.value),
                modality=modality,
                evaluation_dimension=dim,
            ))
        print(f"[Design] Proposed {len(subtasks)} subtasks: {[s.id for s in subtasks]}")
        return subtasks

    def revise(self, subtask: Subtask, reason: str) -> Subtask:
        subtask.description = f"[revised] {subtask.description} ({reason})"
        print(f"[Design] Revised {subtask.id}: {reason}")
        return subtask

    def discard(self, subtask: Subtask, reason: str) -> None:
        print(f"[Design] Discarded {subtask.id}: {reason}")


# ── Grounding Agent ───────────────────────────────────────────────────────────

class GroundingAgent:
    """
    Validates that each subtask can be realized with real data + transformations.
    """

    def search_datasets(self, subtask: Subtask, pool: list[Dataset]) -> list[Dataset]:
        compatible = [d for d in pool if d.modality == subtask.modality or subtask.modality == Modality.OMNI]
        print(f"[Grounding] {subtask.id}: found {len(compatible)} compatible datasets")
        return compatible

    def estimate_transformability(self, subtask: Subtask, dataset: Dataset) -> TransformationPlan:
        # In production: LLM evaluates alignment, robustness, signal preservation
        # Here: mock scores based on modality match
        base = 0.75 if dataset.modality == subtask.modality else 0.50
        return TransformationPlan(
            steps=[f"extract_{subtask.modality.value}", "format_as_mcq", "verify_answer"],
            alignment_score=base + random.uniform(-0.1, 0.1),
            robustness_score=base + random.uniform(-0.1, 0.1),
            signal_score=base + random.uniform(-0.15, 0.1),
        )

    def ground(self, subtask: Subtask, pool: list[Dataset]) -> list[Grounding]:
        candidates = self.search_datasets(subtask, pool)
        groundings = []
        for dataset in candidates:
            plan = self.estimate_transformability(subtask, dataset)
            if plan.is_valid():
                g = Grounding(subtask_id=subtask.id, dataset=dataset, plan=plan)
                groundings.append(g)
                print(f"[Grounding] {subtask.id} ↔ {dataset.name}: score={plan.overall_score:.2f} ✓")
            else:
                print(f"[Grounding] {subtask.id} ↔ {dataset.name}: score={plan.overall_score:.2f} ✗ (below threshold)")
        return groundings


# ── Allocation Agent ──────────────────────────────────────────────────────────

class AllocationAgent:
    """
    Assigns sample quotas across grounded subtasks under global constraints.
    """

    def allocate(self, subtasks: list[Subtask], total_quota: int) -> bool:
        feasible = [s for s in subtasks if s.is_grounded()]
        if not feasible:
            print("[Allocation] No feasible subtasks — allocation failed")
            return False

        per_subtask = total_quota // len(feasible)
        for s in feasible:
            # Pick the best grounding
            best = max(s.groundings, key=lambda g: g.plan.overall_score)
            # Cap by dataset size
            s.quota = min(per_subtask, best.dataset.size // 10)
            print(f"[Allocation] {s.id}: quota={s.quota} from {best.dataset.name} (score={best.plan.overall_score:.2f})")
        return True


# ── Benchmark Executor ────────────────────────────────────────────────────────

class BenchmarkExecutor:
    """
    Realizes the benchmark plan into concrete evaluation items.
    """

    def execute_transformation(self, sample: dict, plan: TransformationPlan) -> dict:
        result = dict(sample)
        for step in plan.steps:
            result[f"step_{step}"] = f"applied:{step}"
        result["question"] = f"Based on the provided content, what is the correct answer?"
        result["answer"] = "A"  # mock
        return result

    def verify(self, item: dict, subtask: Subtask) -> bool:
        # In production: LLM checks semantic validity, format compliance, grounding
        has_question = "question" in item
        has_answer = "answer" in item
        return has_question and has_answer

    def realize(self, subtask: Subtask, max_retries: int = 3) -> list[BenchmarkItem]:
        if not subtask.groundings or subtask.quota == 0:
            return []

        best_grounding = max(subtask.groundings, key=lambda g: g.plan.overall_score)
        items: list[BenchmarkItem] = []
        attempts = 0

        while len(items) < subtask.quota and attempts < subtask.quota * max_retries:
            attempts += 1
            raw = best_grounding.dataset.sample()
            transformed = self.execute_transformation(raw, best_grounding.plan)

            if self.verify(transformed, subtask):
                items.append(BenchmarkItem(
                    subtask_id=subtask.id,
                    question=transformed["question"],
                    answer=transformed["answer"],
                    context=raw,
                    metadata={"dataset": best_grounding.dataset.name, "plan_score": best_grounding.plan.overall_score},
                    verified=True,
                ))

        print(f"[Executor] {subtask.id}: produced {len(items)}/{subtask.quota} items in {attempts} attempts")
        return items


# ── BenchmarkAgent orchestrator ───────────────────────────────────────────────

class BenchmarkAgent:
    def __init__(self, dataset_pool: list[Dataset], total_quota: int = 50):
        self.pool = dataset_pool
        self.total_quota = total_quota
        self.design = DesignAgent()
        self.grounding = GroundingAgent()
        self.allocation = AllocationAgent()
        self.executor = BenchmarkExecutor()

    def run(self, requirement: str, modalities: list[Modality]) -> list[BenchmarkItem]:
        print(f"\n{'='*60}\nRequirement: {requirement}\n{'='*60}")

        # Phase 1: Planner
        subtasks = self.design.propose(requirement, modalities)

        # Ground each subtask
        for s in subtasks:
            s.groundings = self.grounding.ground(s, self.pool)

        # Feasibility gate: every subtask must have at least one grounding
        infeasible = [s for s in subtasks if not s.is_grounded()]
        if infeasible:
            print(f"[Planner] {len(infeasible)} infeasible subtasks — revising...")
            for s in infeasible:
                self.design.revise(s, "no valid grounding found — broadening scope")
                s.groundings = self.grounding.ground(s, self.pool)

        feasible = [s for s in subtasks if s.is_grounded()]
        if not feasible:
            raise RuntimeError("No feasible subtasks after revision — cannot build benchmark")

        ok = self.allocation.allocate(feasible, self.total_quota)
        if not ok:
            raise RuntimeError("Allocation failed")

        # Phase 2: Executor
        all_items: list[BenchmarkItem] = []
        for s in feasible:
            items = self.executor.realize(s)
            all_items.extend(items)

        print(f"\n[BenchmarkAgent] Done — {len(all_items)} items across {len(feasible)} subtasks")
        return all_items


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Build a small dataset pool
    pool = [
        Dataset("MedQA", Modality.TEXT, 1000, "medical", [
            {"id": f"med_{i}", "content": f"Patient presents with symptom_{i}"} for i in range(20)
        ]),
        Dataset("ArtCaption", Modality.IMAGE, 500, "art", [
            {"id": f"art_{i}", "content": f"Artwork depicting scene_{i}"} for i in range(10)
        ]),
        Dataset("PodcastQA", Modality.AUDIO, 300, "general", [
            {"id": f"pod_{i}", "content": f"Audio conversation segment_{i}"} for i in range(10)
        ]),
        Dataset("WikiText", Modality.TEXT, 2000, "general", [
            {"id": f"wiki_{i}", "content": f"Wikipedia passage about topic_{i}"} for i in range(30)
        ]),
    ]

    agent = BenchmarkAgent(pool, total_quota=30)

    # Test 1: text-only benchmark
    print("\n" + "="*60)
    print("TEST 1: Text comprehension benchmark")
    items1 = agent.run(
        "Evaluate a model's ability to reason over medical text and answer clinical questions",
        modalities=[Modality.TEXT],
    )

    # Test 2: multimodal benchmark
    print("\n" + "="*60)
    print("TEST 2: Multimodal understanding benchmark")
    items2 = agent.run(
        "Evaluate a model's understanding of image and audio content together",
        modalities=[Modality.IMAGE, Modality.AUDIO],
    )

    print(f"\nFinal benchmarks: {len(items1)} text items, {len(items2)} multimodal items")
    print(f"Sample item: {items1[0].question!r} → {items1[0].answer!r}")
