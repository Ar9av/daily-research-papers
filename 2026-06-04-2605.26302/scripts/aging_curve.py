"""
Aging curve + half-life for a long-lived agent.

Paper: "Your Agents Are Aging Too: Agent Lifespan Engineering for Deployed Systems"
arXiv: 2605.26302  (https://agingbench.github.io/)

Core idea: a deployed agent's reliability is a curve over its operational lifespan,
not a single day-one score. Run the agent across N sessions under a compaction policy,
score each session, and summarize the curve with half-life and decay slope.

Install: (stdlib only)
Run:     python scripts/aging_curve.py
"""

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class CompactionPolicy:
    """Recompacts the WHOLE memory to a fixed total word budget on every write.

    A fixed budget over a growing history is what creates compression aging: as
    sessions accumulate, the budget can't hold every detail, so the policy decides
    which tokens survive. A naive policy keeps the most recent words and lets old
    low-frequency details (numbers, proper nouns) fall off the end. A careful policy
    preserves digit/Capitalized tokens regardless of age.
    """
    total_budget: int = 24
    careful: bool = False

    @staticmethod
    def _is_detail(tok: str) -> bool:
        return any(c.isdigit() for c in tok) or tok[:1].isupper()

    def recompact(self, store_words: list[str]) -> list[str]:
        if len(store_words) <= self.total_budget:
            return store_words
        if self.careful:
            details = [w for w in store_words if self._is_detail(w)]
            filler = [w for w in store_words if not self._is_detail(w)]
            # Keep every detail token, then backfill the most recent filler.
            keep_filler = filler[-(max(0, self.total_budget - len(details))):]
            return details + keep_filler
        # Naive: keep only the most recent words; old details get squeezed out.
        return store_words[-self.total_budget:]


@dataclass
class MemoryAgent:
    """Toy agent: appends each session's fact, then recompacts the whole store."""
    policy: CompactionPolicy
    store_words: list = field(default_factory=list)

    def write(self, fact: str) -> None:
        self.store_words.extend(fact.split())
        self.store_words = self.policy.recompact(self.store_words)

    def answer(self, probe: str, gold_keyword: str) -> bool:
        # "Recall" succeeds if the gold keyword still survives anywhere in memory.
        return any(gold_keyword.lower() in w.lower() for w in self.store_words)


def aging_curve(
    agent: MemoryAgent,
    sessions: list[tuple[str, str, str]],  # (fact_to_write, probe, gold_keyword)
    held_out_idx: int = 0,
) -> dict:
    """Run the session loop; each session writes a new fact, then we re-probe a
    held-out EARLY detail. As later sessions push the store past its budget, that
    early detail may get recompacted away -> the recall curve ages."""
    _, held_probe, held_gold = sessions[held_out_idx]
    scores = []
    for fact, _, _ in sessions:
        agent.write(fact)
        scores.append(1.0 if agent.answer(held_probe, held_gold) else 0.0)
    return {"scores": scores, **curve_stats(scores)}


def curve_stats(scores: list[float]) -> dict:
    n = len(scores)
    baseline = scores[0] if scores else 0.0
    # Half-life: first session whose score <= 50% of the baseline capability.
    half_life = next(
        (t for t, s in enumerate(scores) if baseline > 0 and s <= 0.5 * baseline),
        float("inf"),
    )
    # Decay slope via ordinary least squares.
    if n > 1:
        xs = list(range(n))
        mx, my = sum(xs) / n, sum(scores) / n
        denom = sum((x - mx) ** 2 for x in xs) or 1.0
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, scores)) / denom
    else:
        slope = 0.0
    hazard = 1.0 - (sum(scores) / n if n else 0.0)  # per-session failure proxy
    return {"half_life": half_life, "decay_slope": round(slope, 4),
            "hazard_proxy": round(hazard, 3)}


def _demo():
    # Each session writes a fact carrying a low-frequency detail (a dose, a name, a $ amount).
    facts = [
        ("Take 50 mg of metoprolol twice daily for blood pressure", "dose?", "50"),
        ("John Smith manages the enterprise sales team in Austin", "who?", "Smith"),
        ("Premium plan renews until January 2026 at $240 per year", "plan?", "240"),
        ("Therapy appointment recurring every Tuesday at 4 pm sharp", "schedule?", "Tuesday"),
        ("Project budget approved at 18000 dollars for Q3 milestones", "budget?", "18000"),
        ("Allergy noted: penicillin causes a severe rash reaction", "allergy?", "penicillin"),
        ("Flight UA482 departs 6:15 am from gate B12 on Friday", "flight?", "UA482"),
        ("Contractor invoice 7731 due net 30 from acme corp", "invoice?", "7731"),
        ("Diet restriction lifted: gluten is now allowed again", "diet?", "gluten"),
        ("Server key rotation scheduled for the 15th of the month", "rotation?", "15th"),
    ]

    for label, policy in [
        ("lossy  (budget=24, recency)", CompactionPolicy(total_budget=24, careful=False)),
        ("careful(budget=24, keep #/names)", CompactionPolicy(total_budget=24, careful=True)),
    ]:
        result = aging_curve(MemoryAgent(policy), facts)
        bar = "".join("#" if s else "." for s in result["scores"])
        print(f"\n{label}")
        print(f"  recall curve : [{bar}]")
        print(f"  half-life    : {result['half_life']} sessions")
        print(f"  decay slope  : {result['decay_slope']}")
        print(f"  hazard proxy : {result['hazard_proxy']}")
    print("\nSame model, same budget — the write policy alone moves the half-life.")


if __name__ == "__main__":
    _demo()
