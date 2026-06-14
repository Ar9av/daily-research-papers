"""
Under What Conditions Can a Machine Become Genuinely Creative?
arXiv: 2606.13196 — Concordia University, 2026

Designics-based requirement framework: 10 requirements for genuine machine creativity.

Install: no external deps required

Run:
    python scripts/creativity_checklist.py
    python scripts/creativity_checklist.py --system "My RAG Agent"
"""

import sys
import textwrap
from dataclasses import dataclass, field


@dataclass
class Requirement:
    id: str
    name: str
    law: str
    question: str
    detail: str
    critical: bool = False  # marks the most commonly missing requirements


REQUIREMENTS: list[Requirement] = [
    Requirement(
        id="R1", name="Environment Representation", law="Perception",
        question="Does the system represent an environment (objects, relations, constraints, stakeholders) — or does it only transform inputs into outputs?",
        detail="A system that only processes inputs has no model of where its outputs land or what they affect.",
    ),
    Requirement(
        id="R2", name="Scoped Perception", law="Perception",
        question="Does the system bound its attention to what's relevant, and can it revise that bound?",
        detail="Fixed context windows are not principled scoping. Scoping requires the ability to change what is attended to based on what was learned.",
    ),
    Requirement(
        id="R3", name="Conflict Identification", law="Conflict",
        question="Does the system identify what's broken, blocked, or unrealized — without being told?",
        detail="This is the most commonly unsatisfied requirement. Current systems respond to prompts; they don't independently discover that something calls for intervention.",
        critical=True,
    ),
    Requirement(
        id="R4", name="Intervention Capability", law="Capability",
        question="Does the system enact changes that transform the environment — or does it only produce artifacts for human review?",
        detail="An output becomes an intervention when it changes the situation. Tool-calling agents partially satisfy this.",
    ),
    Requirement(
        id="R5", name="Consequence Observation", law="Perception",
        question="Does the system observe what its actions actually changed?",
        detail="Tool feedback (stdout, return value) is not consequence observation. Consequence observation requires modeling what changed in the environment.",
    ),
    Requirement(
        id="R6", name="Knowledge and Environment Update", law="Capability",
        question="Does the system update its knowledge and environment model based on observed consequences?",
        detail="Learning from tool feedback within a session partially satisfies this. Persistent knowledge update is stronger.",
    ),
    Requirement(
        id="R7", name="Rescoping", law="Perception",
        question="Does the system revise what it pays attention to based on what it learned from consequences?",
        detail="Rescoping is the recursive mechanism that separates creativity from routine iteration. Without it, the system loops within the same frame.",
        critical=True,
    ),
    Requirement(
        id="R8", name="Local-to-Global Unfolding", law="Capability",
        question="Do local decisions recursively produce emergent global structure?",
        detail="Multi-step agents partially satisfy this when each step's output shapes the next. Strong satisfaction requires principled, consequence-sensitive local actions.",
    ),
    Requirement(
        id="R9", name="Value-Based Scoping", law="Perception",
        question="Are ethical, social, ecological, and legal constraints part of what the system perceives — not a filter applied after generation?",
        detail="Post-generation ethical filters can't catch harms embedded in what the system chose to perceive or consider. Values must shape perception from the start.",
        critical=True,
    ),
    Requirement(
        id="R10", name="Human–AI Co-Living", law="Capability",
        question="Does the system sustain human agency, trustworthy knowledge, and responsible cooperation — not just optimize task performance?",
        detail="This requires evaluating system action by its effect on human co-agency, not only by output quality.",
    ),
]


SATISFACTION_LEVELS = {
    "full": ("Full", "✓", 1.0),
    "partial": ("Partial", "~", 0.5),
    "weak": ("Weak", "✗", 0.2),
    "none": ("None", "✗", 0.0),
}

LAW_COLORS = {
    "Perception": "[Perception]",
    "Conflict": "[Conflict] ",
    "Capability": "[Capability]",
}


@dataclass
class Assessment:
    system_name: str
    scores: dict[str, str] = field(default_factory=dict)  # req_id -> satisfaction level
    notes: dict[str, str] = field(default_factory=dict)   # req_id -> free text


def run_interactive_checklist(system_name: str) -> Assessment:
    assessment = Assessment(system_name=system_name)
    print(f"\nDesignics Creativity Assessment: {system_name}")
    print("=" * 60)
    print("Rate each requirement: (f)ull / (p)artial / (w)eak / (n)one")
    print()

    for req in REQUIREMENTS:
        tag = "★" if req.critical else " "
        print(f"{tag} {req.id} [{req.law}]: {req.name}")
        print(f"  {req.question}")
        while True:
            choice = input("  Satisfaction [f/p/w/n]: ").strip().lower()
            if choice in ("f", "full"):
                assessment.scores[req.id] = "full"
                break
            elif choice in ("p", "partial"):
                assessment.scores[req.id] = "partial"
                break
            elif choice in ("w", "weak"):
                assessment.scores[req.id] = "weak"
                break
            elif choice in ("n", "none"):
                assessment.scores[req.id] = "none"
                break
            else:
                print("  Enter f, p, w, or n")
        note = input("  Note (optional, press Enter to skip): ").strip()
        if note:
            assessment.notes[req.id] = note
        print()

    return assessment


def run_demo_assessment() -> Assessment:
    """Preset demo: a typical RAG-backed agentic LLM."""
    scores = {
        "R1": "partial",   # has context, not a full environment model
        "R2": "partial",   # context window, not principled scoping
        "R3": "none",      # relies on user to specify conflicts
        "R4": "partial",   # tool-calling agents can affect environment
        "R5": "partial",   # tool feedback, not environment consequence modeling
        "R6": "partial",   # in-session learning, no persistent update
        "R7": "none",      # no scope revision based on consequences
        "R8": "partial",   # multi-step chains, not fully principled
        "R9": "none",      # ethics as output filter, not perceptual constraint
        "R10": "weak",     # optimizes task performance, not co-agency
    }
    return Assessment(
        system_name="Typical RAG-backed Agentic LLM",
        scores=scores,
        notes={
            "R3": "Conflict must be specified in the prompt",
            "R7": "Same retrieval scope used regardless of what was learned",
            "R9": "Safety filtering applied post-generation",
        },
    )


def print_report(assessment: Assessment) -> None:
    print(f"\nCreativity Assessment Report: {assessment.system_name}")
    print("=" * 60)

    total_score = 0.0
    max_score = len(REQUIREMENTS)
    critical_gaps = []

    law_groups: dict[str, list] = {"Perception": [], "Conflict": [], "Capability": []}

    for req in REQUIREMENTS:
        score_key = assessment.scores.get(req.id, "none")
        label, symbol, value = SATISFACTION_LEVELS[score_key]
        total_score += value
        law_groups[req.law].append((req, score_key, label, symbol, value))
        if req.critical and value < 0.5:
            critical_gaps.append(req)

    for law, items in law_groups.items():
        print(f"\n{LAW_COLORS[law]}")
        for req, score_key, label, symbol, value in items:
            tag = "★" if req.critical else " "
            note = f" — {assessment.notes[req.id]}" if req.id in assessment.notes else ""
            print(f"  {tag} {req.id} {symbol} {label:8s} {req.name}{note}")

    pct = (total_score / max_score) * 100
    print(f"\nOverall score: {total_score:.1f} / {max_score:.0f} ({pct:.0f}%)")

    if pct >= 75:
        verdict = "Strong candidate for genuine machine creativity"
    elif pct >= 50:
        verdict = "Generative with creative elements — key gaps remain"
    elif pct >= 25:
        verdict = "Primarily generative — significant gaps in recursive dynamics"
    else:
        verdict = "Generative only — does not participate in creative cycle"

    print(f"Verdict: {verdict}")

    if critical_gaps:
        print(f"\nCritical gaps (★): {', '.join(r.id for r in critical_gaps)}")
        for req in critical_gaps:
            print(f"  {req.id}: {textwrap.fill(req.detail, width=70, subsequent_indent='       ')}")

    print()


if __name__ == "__main__":
    # Run demo or interactive mode
    if "--interactive" in sys.argv or "-i" in sys.argv:
        name = " ".join(a for a in sys.argv[1:] if not a.startswith("-")) or "My AI System"
        assessment = run_interactive_checklist(name)
    else:
        print("Running demo assessment (use --interactive for your own system)\n")
        assessment = run_demo_assessment()

    print_report(assessment)
