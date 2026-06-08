"""
OPENSKILL: Open-World Self-Evolution for LLM Agents
arxiv: 2606.06741  —  Yan et al., Lehigh / UIC / Salesforce, 2026

Install: pip install anthropic  (optional — demo runs with mocks)
Run:     python scripts/openskill.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Domain types ──────────────────────────────────────────────────────────────

@dataclass
class KnowledgeDoc:
    source: str
    content: str
    is_anchor: bool = False     # True = verification anchor (checkable fact)


@dataclass
class VerificationAnchor:
    description: str
    check: str                  # e.g. "output has 42 rows", "metric in [0, 1]"
    expected_value: Any


@dataclass
class VirtualTest:
    name: str
    anchor: VerificationAnchor

    def run(self, result: dict) -> bool:
        # In production: execute as a pytest assertion
        # Here: simulate with a mock check
        return result.get(self.anchor.check) == self.anchor.expected_value


@dataclass
class Skill:
    task: str
    plan: str
    knowledge_sources: list[str]
    implementation: str
    version: int = 0


@dataclass
class FailureDiagnostic:
    failed_tests: list[str]
    root_cause: str
    is_knowledge_gap: bool
    gap_query: str = ""
    revision_suggestion: str = ""


# ── Open-World Retriever ──────────────────────────────────────────────────────

class OpenWorldRetriever:
    """
    Simulates querying documentation, repos, and the web.
    In production: real web search + repo indexing.
    """

    MOCK_KNOWLEDGE: dict[str, list[KnowledgeDoc]] = {
        "data_analysis": [
            KnowledgeDoc("pandas_docs", "DataFrame.groupby aggregates rows by key. Use .agg() for multi-column stats."),
            KnowledgeDoc("numpy_docs", "np.corrcoef returns Pearson correlation matrix. Values in [-1, 1]."),
        ],
        "file_processing": [
            KnowledgeDoc("python_stdlib", "pathlib.Path.glob returns generator of matching paths."),
            KnowledgeDoc("csv_spec", "RFC 4180: CSV fields separated by commas, quoted if containing commas."),
        ],
        "api_usage": [
            KnowledgeDoc("requests_docs", "requests.get returns Response. .json() parses body as JSON."),
            KnowledgeDoc("httpx_docs", "httpx.Client supports connection pooling; use as context manager."),
        ],
    }

    MOCK_ANCHORS: dict[str, list[VerificationAnchor]] = {
        "data_analysis": [
            VerificationAnchor("Iris dataset has 150 rows", "row_count", 150),
            VerificationAnchor("Correlation values in [-1, 1]", "metric_range_valid", True),
        ],
        "file_processing": [
            VerificationAnchor("Output file exists", "output_exists", True),
            VerificationAnchor("CSV has header row", "has_header", True),
        ],
        "api_usage": [
            VerificationAnchor("Response status 200", "status_code", 200),
            VerificationAnchor("Response body is valid JSON", "is_json", True),
        ],
    }

    def _infer_domain(self, task: str) -> str:
        for domain in self.MOCK_KNOWLEDGE:
            if any(kw in task.lower() for kw in domain.split("_")):
                return domain
        return "data_analysis"

    def retrieve_knowledge(self, task: str) -> list[KnowledgeDoc]:
        domain = self._infer_domain(task)
        docs = self.MOCK_KNOWLEDGE.get(domain, [])
        print(f"  [Retriever] Retrieved {len(docs)} knowledge docs for '{domain}'")
        return docs

    def retrieve_anchors(self, task: str) -> list[VerificationAnchor]:
        domain = self._infer_domain(task)
        anchors = self.MOCK_ANCHORS.get(domain, [])
        print(f"  [Retriever] Retrieved {len(anchors)} verification anchors")
        return anchors

    def retrieve_targeted(self, gap_query: str) -> list[KnowledgeDoc]:
        print(f"  [Retriever:Gap] Targeted retrieval for: '{gap_query}'")
        return [KnowledgeDoc("targeted_result", f"Specific knowledge for: {gap_query}")]


# ── Virtual Verifier ──────────────────────────────────────────────────────────

class VirtualVerifier:
    """
    Synthesizes a deterministic test suite from verification anchors.
    Never accesses GT tests.
    """

    def generate_tests(self, anchors: list[VerificationAnchor]) -> list[VirtualTest]:
        tests = [VirtualTest(name=f"test_{a.description[:30].replace(' ', '_')}", anchor=a)
                 for a in anchors]
        # Virtual verifier generates ~3.4x more tests than GT — add defensive checks
        extra_checks = [
            VirtualTest("test_output_not_empty",
                        VerificationAnchor("Output is non-empty", "output_not_empty", True)),
            VirtualTest("test_no_exception",
                        VerificationAnchor("No exception raised", "no_exception", True)),
            VirtualTest("test_output_type_valid",
                        VerificationAnchor("Output type is valid", "output_type_valid", True)),
        ]
        all_tests = tests + extra_checks
        print(f"  [Verifier] Generated {len(all_tests)} virtual tests ({len(tests)} anchored + {len(extra_checks)} defensive)")
        return all_tests

    def run_tests(self, tests: list[VirtualTest], result: dict) -> tuple[float, list[str]]:
        failures = []
        for t in tests:
            if not t.run(result):
                failures.append(t.name)
        pass_rate = 1.0 - len(failures) / len(tests)
        return pass_rate, failures

    def diagnose(self, failures: list[str], skill: Skill) -> FailureDiagnostic:
        is_gap = any("knowledge" in f.lower() or "anchor" in f.lower() for f in failures)
        return FailureDiagnostic(
            failed_tests=failures,
            root_cause="missing domain knowledge" if is_gap else "implementation bug",
            is_knowledge_gap=is_gap,
            gap_query=failures[0].replace("test_", "").replace("_", " ") if is_gap else "",
            revision_suggestion=f"Fix: {failures[0]}" if failures else "",
        )


# ── OpenSkill Pipeline ────────────────────────────────────────────────────────

class OpenSkillPipeline:
    def __init__(self, J: int = 3):
        self.retriever = OpenWorldRetriever()
        self.verifier = VirtualVerifier()
        self.J = J

    def _simulate_execution(self, skill: Skill, inject_failure: bool = False) -> dict:
        """Simulate agent execution. In production: real sandbox execution."""
        result = {
            "row_count": 150 if not inject_failure else 0,
            "metric_range_valid": True,
            "output_exists": True,
            "has_header": True,
            "status_code": 200,
            "is_json": True,
            "output_not_empty": True,
            "no_exception": True,
            "output_type_valid": True,
        }
        if inject_failure:
            result["row_count"] = 0
            result["metric_range_valid"] = False
        return result

    def run(self, task: str, inject_initial_failure: bool = True) -> tuple[Skill, float]:
        print(f"\n{'='*60}")
        print(f"[OpenSkill] Task: '{task}'")
        print(f"{'='*60}")

        # Stage 1: Open-world knowledge acquisition
        print("\n[Stage 1] Open-world knowledge acquisition")
        knowledge = self.retriever.retrieve_knowledge(task)
        anchors = self.retriever.retrieve_anchors(task)

        plan = f"Skill plan for: {task}\n" + "\n".join(f"  - {d.content[:60]}" for d in knowledge)
        skill = Skill(task=task, plan=plan,
                      knowledge_sources=[d.source for d in knowledge],
                      implementation=f"# Initial skill for: {task}", version=0)

        # Stage 2: Leakage-free evolution
        print("\n[Stage 2] Leakage-free skill evolution (GT tests hidden)")
        virtual_tests = self.verifier.generate_tests(anchors)

        for j in range(self.J):
            # Simulate: first round fails if inject_initial_failure, later rounds succeed
            result = self._simulate_execution(skill, inject_failure=(inject_initial_failure and j == 0))
            pass_rate, failures = self.verifier.run_tests(virtual_tests, result)
            print(f"  Round {j+1}: proxy pass rate = {pass_rate:.1%}, failures = {failures[:2]}")

            if pass_rate == 1.0:
                print(f"  [Evolution] All virtual tests pass — terminating at round {j+1}")
                break

            diagnostic = self.verifier.diagnose(failures, skill)
            print(f"  [Diagnostic] root_cause='{diagnostic.root_cause}', is_gap={diagnostic.is_knowledge_gap}")

            if diagnostic.is_knowledge_gap:
                extra = self.retriever.retrieve_targeted(diagnostic.gap_query)
                knowledge += extra

            skill = Skill(task=task, plan=plan,
                          knowledge_sources=[d.source for d in knowledge],
                          implementation=f"# Refined skill v{j+1} for: {task}",
                          version=j + 1)

        # Stage 3: zero-shot deployment (GT unlocked here)
        print("\n[Stage 3] Zero-shot deployment — GT evaluation")
        final_result = self._simulate_execution(skill, inject_failure=False)
        gt_pass = float(final_result.get("row_count", 0) > 0)  # mock GT test
        print(f"  GT pass rate: {gt_pass:.1%}")

        return skill, gt_pass


# ── Cross-model transfer demo ─────────────────────────────────────────────────

def transfer_demo(skill: Skill, target_models: list[str]) -> None:
    print(f"\n{'='*60}")
    print("Cross-model transfer (same skill file, no adaptation)")
    print(f"{'='*60}")
    for model in target_models:
        # Simulate performance — OpenSkill skills are model-agnostic
        gain = random.uniform(5.5, 14.8)
        print(f"  {model}: +{gain:.1f}pp over no-skill baseline")


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = OpenSkillPipeline(J=3)

    # Task 1: data analysis
    skill1, score1 = pipeline.run("Analyze a CSV dataset and compute correlation statistics")

    # Task 2: API usage
    skill2, score2 = pipeline.run("Fetch data from a REST API and process the JSON response")

    print(f"\nFinal scores: {score1:.1%}, {score2:.1%}")

    # Cross-model transfer
    transfer_demo(skill1, ["Haiku 4.5", "Qwen 3 Coder", "DeepSeek V3", "Mistral Large 3"])
