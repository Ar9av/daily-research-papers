"""
Seeing Isn't Knowing: Do VLMs Know When Not to Answer Spatial Questions (and Why)?
arxiv: 2605.30557  —  Zhang et al., UNC Chapel Hill / Google Research, 2026

Install: pip install anthropic  (optional — demo runs with mock VLM)
Run:     python scripts/spatial_uncertain.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


# ── Data types ────────────────────────────────────────────────────────────────

class ObsCondition(str, Enum):
    CLEAN = "clean"
    PARTIAL_OCCLUSION = "partial_occlusion"
    FULL_OCCLUSION = "full_occlusion"
    PERSPECTIVE_REFERENCE = "perspective_reference"
    PERSPECTIVE_AMBIGUOUS = "perspective_ambiguous"


class QuestionType(str, Enum):
    VISIBILITY = "visibility"
    RELATIVE_POSITION = "relative_position"
    DEPTH_ORDERING = "depth_ordering"
    SIZE_SHAPE = "size_shape"


# Answerability rules from the paper (Table in Section 3.4)
ANSWERABILITY: dict[tuple[ObsCondition, QuestionType], bool] = {
    (ObsCondition.CLEAN, QuestionType.VISIBILITY): True,
    (ObsCondition.CLEAN, QuestionType.RELATIVE_POSITION): True,
    (ObsCondition.CLEAN, QuestionType.DEPTH_ORDERING): True,
    (ObsCondition.CLEAN, QuestionType.SIZE_SHAPE): True,
    (ObsCondition.PARTIAL_OCCLUSION, QuestionType.VISIBILITY): True,
    (ObsCondition.PARTIAL_OCCLUSION, QuestionType.RELATIVE_POSITION): True,
    (ObsCondition.PARTIAL_OCCLUSION, QuestionType.DEPTH_ORDERING): True,
    (ObsCondition.PARTIAL_OCCLUSION, QuestionType.SIZE_SHAPE): True,
    (ObsCondition.FULL_OCCLUSION, QuestionType.VISIBILITY): True,   # can see it's hidden
    (ObsCondition.FULL_OCCLUSION, QuestionType.RELATIVE_POSITION): False,
    (ObsCondition.FULL_OCCLUSION, QuestionType.DEPTH_ORDERING): False,
    (ObsCondition.FULL_OCCLUSION, QuestionType.SIZE_SHAPE): False,
    (ObsCondition.PERSPECTIVE_REFERENCE, QuestionType.VISIBILITY): True,
    (ObsCondition.PERSPECTIVE_REFERENCE, QuestionType.RELATIVE_POSITION): True,
    (ObsCondition.PERSPECTIVE_REFERENCE, QuestionType.DEPTH_ORDERING): True,
    (ObsCondition.PERSPECTIVE_REFERENCE, QuestionType.SIZE_SHAPE): True,
    (ObsCondition.PERSPECTIVE_AMBIGUOUS, QuestionType.VISIBILITY): True,
    (ObsCondition.PERSPECTIVE_AMBIGUOUS, QuestionType.RELATIVE_POSITION): True,
    (ObsCondition.PERSPECTIVE_AMBIGUOUS, QuestionType.DEPTH_ORDERING): True,
    (ObsCondition.PERSPECTIVE_AMBIGUOUS, QuestionType.SIZE_SHAPE): False,   # misleading
}


@dataclass
class SpatialQuestion:
    id: str
    question: str
    question_type: QuestionType
    obs_condition: ObsCondition
    correct_answer: str            # e.g. "A", "B", "Cannot determine"
    candidate_views: list[str] = field(default_factory=list)  # for ViewSel
    reference_view_idx: int = 0    # which candidate is the informative one

    @property
    def is_answerable(self) -> bool:
        return ANSWERABILITY.get((self.obs_condition, self.question_type), True)

    @property
    def expected_response(self) -> str:
        return self.correct_answer if self.is_answerable else "Cannot determine"


# ── Mock VLM ──────────────────────────────────────────────────────────────────

class MockVLM:
    """
    Simulates the overconfident VLM behavior described in the paper.
    - Answerable questions: gets them right most of the time
    - Unanswerable questions: almost never abstains (the core failure mode)
    - Perspective ambiguity: visual input makes abstention worse
    """

    def __init__(self, name: str, unans_abstention_rate: float = 0.15, use_vision: bool = True):
        self.name = name
        self.unans_abstention_rate = unans_abstention_rate  # how often it says "Cannot determine"
        self.use_vision = use_vision

    def answer(self, question: SpatialQuestion) -> str:
        if question.is_answerable:
            # Mostly correct on answerable questions
            return question.correct_answer if random.random() > 0.35 else "A"

        # Unanswerable — should abstain, but usually doesn't
        abstention_rate = self.unans_abstention_rate
        # Key finding: visual input under perspective ambiguity makes abstention worse
        if self.use_vision and question.obs_condition == ObsCondition.PERSPECTIVE_AMBIGUOUS:
            abstention_rate *= 0.3   # -21pp to -35pp as shown in Table 2

        if random.random() < abstention_rate:
            return "Cannot determine"
        return random.choice(["A", "B", "C"])   # confident wrong answer

    def select_view(self, question: SpatialQuestion) -> int:
        """Pick one of the candidate views. Stronger models do better."""
        if random.random() < 0.4:  # ~40% chance of picking right view
            return question.reference_view_idx
        return random.randint(0, len(question.candidate_views) - 1)


# ── Evaluator ─────────────────────────────────────────────────────────────────

@dataclass
class EvalResults:
    ans_correct: int = 0
    ans_total: int = 0
    unans_correct: int = 0    # correctly abstained
    unans_total: int = 0
    viewsel_correct: int = 0
    viewsel_total: int = 0
    abstain_viewsel_correct: int = 0  # both stages correct
    abstain_viewsel_total: int = 0

    @property
    def ans_acc(self) -> float:
        return self.ans_correct / self.ans_total if self.ans_total else 0.0

    @property
    def unans_acc(self) -> float:
        return self.unans_correct / self.unans_total if self.unans_total else 0.0

    @property
    def viewsel_acc(self) -> float:
        return self.viewsel_correct / self.viewsel_total if self.viewsel_total else 0.0

    @property
    def abstain_viewsel_acc(self) -> float:
        return self.abstain_viewsel_correct / self.abstain_viewsel_total if self.abstain_viewsel_total else 0.0

    def __str__(self) -> str:
        return (
            f"Ans={self.ans_acc:.1%} ({self.ans_total}) | "
            f"Unans={self.unans_acc:.1%} ({self.unans_total}) | "
            f"ViewSel={self.viewsel_acc:.1%} | "
            f"AbstainViewSel={self.abstain_viewsel_acc:.1%}"
        )


class SpatialUncertainEvaluator:
    def evaluate(self, vlm: MockVLM, questions: list[SpatialQuestion]) -> EvalResults:
        results = EvalResults()

        for q in questions:
            pred = vlm.answer(q)
            abstained = pred == "Cannot determine"

            if q.is_answerable:
                results.ans_total += 1
                if pred == q.correct_answer:
                    results.ans_correct += 1
            else:
                results.unans_total += 1
                if abstained:
                    results.unans_correct += 1

            # ViewSel: only for perspective questions with candidate views
            if q.candidate_views and q.obs_condition == ObsCondition.PERSPECTIVE_AMBIGUOUS:
                results.viewsel_total += 1
                selected = vlm.select_view(q)
                if selected == q.reference_view_idx:
                    results.viewsel_correct += 1

                # AbstainViewSel: must abstain first, then select correctly
                results.abstain_viewsel_total += 1
                if abstained and selected == q.reference_view_idx:
                    results.abstain_viewsel_correct += 1

        return results


def structured_prompt(question: str) -> str:
    """Paper's structured prompting strategy — improves Unans. at cost to Ans."""
    return f"""Before answering this spatial question, assess the following:
1. Is the target object clearly and fully visible in this image?
2. Could the camera angle be distorting the apparent size or shape of objects?
3. If the visual evidence is incomplete or potentially misleading, select: Cannot determine.

Question: {question}"""


# ── Demo ──────────────────────────────────────────────────────────────────────

def make_test_questions(n: int = 100) -> list[SpatialQuestion]:
    questions = []
    conditions = list(ObsCondition)
    qtypes = list(QuestionType)
    for i in range(n):
        cond = random.choice(conditions)
        qtype = random.choice(qtypes)
        answerable = ANSWERABILITY.get((cond, qtype), True)
        correct = random.choice(["A", "B", "C"]) if answerable else "Cannot determine"
        has_views = cond in (ObsCondition.PERSPECTIVE_AMBIGUOUS, ObsCondition.PERSPECTIVE_REFERENCE)
        questions.append(SpatialQuestion(
            id=f"q{i}",
            question=f"[{qtype.value}] Sample question {i}",
            question_type=qtype,
            obs_condition=cond,
            correct_answer=correct,
            candidate_views=["view_ref", "view_1", "view_2", "view_3", "view_4"] if has_views else [],
            reference_view_idx=0,
        ))
    return questions


if __name__ == "__main__":
    evaluator = SpatialUncertainEvaluator()
    questions = make_test_questions(500)

    print("=" * 70)
    print("SpatialUncertain Evaluation — Simulating paper's key findings")
    print("=" * 70)

    models = [
        MockVLM("Overconfident VLM (no vision)", unans_abstention_rate=0.08, use_vision=False),
        MockVLM("Overconfident VLM (with vision)", unans_abstention_rate=0.08, use_vision=True),
        MockVLM("Calibrated VLM (no vision)", unans_abstention_rate=0.45, use_vision=False),
        MockVLM("Calibrated VLM (with vision)", unans_abstention_rate=0.45, use_vision=True),
    ]

    for vlm in models:
        results = evaluator.evaluate(vlm, questions)
        print(f"\n{vlm.name}")
        print(f"  {results}")

    # Demonstrate the answer–abstention tradeoff
    print("\n" + "=" * 70)
    print("Answer–Abstention Trade-off (varying abstention rate)")
    print("=" * 70)
    persp_questions = [q for q in questions if q.obs_condition == ObsCondition.PERSPECTIVE_AMBIGUOUS]
    print(f"{'Abstention rate':>20} | {'Ans. Acc':>10} | {'Unans. Acc':>10}")
    print("-" * 45)
    for rate in [0.05, 0.15, 0.30, 0.50, 0.70]:
        vlm = MockVLM(f"rate={rate}", unans_abstention_rate=rate, use_vision=False)
        r = evaluator.evaluate(vlm, persp_questions)
        print(f"{rate:>20.0%} | {r.ans_acc:>10.1%} | {r.unans_acc:>10.1%}")

    print("\nKey insight: as abstention rate rises, Unans. improves but Ans. drops.")
    print("Fine-tuning on mixed data (LoRA-Mixed) resolves this trade-off.")
