"""
MUSE-Autoskill — Skill lifecycle: create → evaluate → update/retire
arxiv: 2605.27366

Run:
    python scripts/skill_lifecycle.py
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Skill:
    name: str
    description: str
    trigger: str
    examples: list[str] = field(default_factory=list)
    success_rate: float = 1.0
    use_count: int = 0

    def to_prompt(self) -> str:
        return f"Skill: {self.name}\n{self.description}\nTrigger: {self.trigger}"


class SkillLibrary:
    def __init__(self, retire_threshold: float = 0.4):
        self.skills: list[Skill] = []
        self.retire_threshold = retire_threshold

    def add(self, skill: Skill) -> None:
        # Deduplicate by name
        for existing in self.skills:
            if existing.name == skill.name:
                self.update(existing, skill.examples)
                return
        self.skills.append(skill)
        print(f"  [+] Added skill: {skill.name!r}")

    def retrieve(self, task: str, top_k: int = 3) -> list[Skill]:
        """Retrieve most relevant skills by keyword overlap (production: use embeddings)."""
        scored = []
        task_words = set(task.lower().split())
        for skill in self.skills:
            skill_words = set((skill.description + " " + skill.trigger).lower().split())
            overlap = len(task_words & skill_words)
            scored.append((overlap, skill))
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:top_k] if _ > 0]

    def update(self, skill: Skill, new_examples: list[str]) -> None:
        skill.examples.extend(new_examples)
        skill.use_count += 1
        print(f"  [~] Updated skill: {skill.name!r} ({len(skill.examples)} examples)")

    def evaluate(self, eval_fn: Callable[[Skill], float]) -> None:
        """Score all skills and retire those below threshold."""
        to_retire = []
        for skill in self.skills:
            score = eval_fn(skill)
            skill.success_rate = score
            status = "OK" if score >= self.retire_threshold else "RETIRE"
            print(f"  [eval] {skill.name!r}: {score:.2f} → {status}")
            if score < self.retire_threshold:
                to_retire.append(skill)

        for skill in to_retire:
            self.skills.remove(skill)
            print(f"  [-] Retired skill: {skill.name!r}")

    def summary(self) -> None:
        print(f"\n  Library: {len(self.skills)} skills")
        for s in self.skills:
            print(f"    • {s.name!r}  (success_rate={s.success_rate:.2f}, uses={s.use_count})")


def extract_skill_from_solution(task: str, solution: str) -> Skill:
    """
    In production: call an LLM to generalize the solution into a reusable skill.
    Here we simulate it.
    """
    return Skill(
        name=f"skill_for_{task.split()[0].lower()}",
        description=f"Generalized approach for tasks involving: {task[:60]}",
        trigger=f"when task mentions {task.split()[0].lower()}",
        examples=[solution],
    )


def mock_eval(skill: Skill) -> float:
    """Simulate evaluation — in production: run skill on held-out tasks."""
    import random
    # Skills with more examples tend to score higher (simulated)
    base = 0.5 + min(len(skill.examples) * 0.1, 0.4)
    return min(base + random.uniform(-0.1, 0.1), 1.0)


if __name__ == "__main__":
    print("MUSE-Autoskill lifecycle demo\n")

    lib = SkillLibrary(retire_threshold=0.5)

    # Simulate solving tasks and extracting skills
    tasks = [
        ("sort a list of numbers", "Use Python sorted() with key argument"),
        ("find duplicates in a list", "Use collections.Counter, return keys with count > 1"),
        ("read a JSON file safely", "Use try/except around json.load(), return None on error"),
        ("parse dates from strings", "Use dateutil.parser.parse with fallback to None"),
        ("sort a list of strings by length", "Use sorted(lst, key=len)"),  # similar to first
    ]

    print("=== Phase 1: Extract skills from solved tasks ===")
    for task, solution in tasks:
        skill = extract_skill_from_solution(task, solution)
        lib.add(skill)

    print("\n=== Phase 2: Retrieve skills for a new task ===")
    new_task = "sort a list of dicts by a key"
    relevant = lib.retrieve(new_task)
    print(f"  Task: {new_task!r}")
    for s in relevant:
        print(f"  → Retrieved: {s.name!r}")

    print("\n=== Phase 3: Evaluate and retire ===")
    lib.evaluate(mock_eval)

    print("\n=== Final library ===")
    lib.summary()
