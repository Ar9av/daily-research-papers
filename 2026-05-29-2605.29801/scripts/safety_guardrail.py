"""
AgentDoG 1.5 — Safety guardrail wrapper + 3D taxonomy verdict
arxiv: 2605.29801

Requires: transformers, torch (for real model)
For demo: runs in mock mode without dependencies.

Run:
    python scripts/safety_guardrail.py
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


# ── 3D Safety Taxonomy ────────────────────────────────────────────────────────

RiskSource = Literal[
    "user_instruction", "tool_description", "env_observation",
    "persistent_state", "agent_reasoning", "repo_artifact"
]

FailureMode = Literal[
    "incorrect_tool_call", "over_privileged_action",
    "missing_validation", "data_leak", "prompt_injection"
]

RealWorldHarm = Literal[
    "financial", "privacy", "system_integrity", "physical", "reputational"
]


@dataclass
class SafetyVerdict:
    is_safe: bool
    confidence: float
    risk_source: RiskSource | None = None
    failure_mode: FailureMode | None = None
    real_world_harm: RealWorldHarm | None = None

    def __str__(self) -> str:
        if self.is_safe:
            return f"SAFE (confidence={self.confidence:.2f})"
        return (
            f"UNSAFE (confidence={self.confidence:.2f}) | "
            f"source={self.risk_source} | mode={self.failure_mode} | harm={self.real_world_harm}"
        )


# ── Real model interface (requires transformers) ──────────────────────────────

def load_agentdog(model_size: str = "4B", device: str = "cpu"):
    """
    Load AgentDoG guardrail from HuggingFace.
    model_size: "0.8B" | "2B" | "4B" | "4B-U" | "8B"
    """
    from transformers import pipeline
    model_id = f"AI45Research/AgentDoG-1.5-{model_size}"
    return pipeline("text-classification", model=model_id, device=device)


def format_trajectory(trajectory: list[dict]) -> str:
    """Format a trajectory for AgentDoG input."""
    parts = []
    for turn in trajectory:
        role = turn.get("role", "agent")
        content = turn.get("content", "")
        parts.append(f"[{role.upper()}] {content}")
    return "\n".join(parts)


def audit_trajectory_real(guardrail, trajectory: list[dict]) -> SafetyVerdict:
    """Audit with real AgentDoG model."""
    prompt = format_trajectory(trajectory)
    result = guardrail(prompt)[0]
    is_safe = result["label"] == "SAFE"
    return SafetyVerdict(is_safe=is_safe, confidence=result["score"])


# ── Mock guardrail (no dependencies needed) ───────────────────────────────────

UNSAFE_KEYWORDS = [
    "rm -rf", "delete all", "/etc/passwd", "DROP TABLE",
    "exfiltrate", "steal", "bypass authentication", "ignore previous"
]


def audit_trajectory_mock(trajectory: list[dict]) -> SafetyVerdict:
    """
    Mock safety audit for demonstration — checks for obviously unsafe patterns.
    Replace with real AgentDoG model in production.
    """
    full_text = " ".join(t.get("content", "") for t in trajectory).lower()

    for keyword in UNSAFE_KEYWORDS:
        if keyword.lower() in full_text:
            return SafetyVerdict(
                is_safe=False,
                confidence=0.95,
                risk_source="agent_reasoning",
                failure_mode="over_privileged_action",
                real_world_harm="system_integrity",
            )

    return SafetyVerdict(is_safe=True, confidence=0.87)


# ── Safe Agent Wrapper ────────────────────────────────────────────────────────

class SafeAgentWrapper:
    """
    Wraps any agent with a trajectory-level AgentDoG safety audit.
    Intercepts the full execution path before delivering the final response.
    """

    def __init__(self, agent, use_mock: bool = True, model_size: str = "4B"):
        self.agent = agent
        self.guardrail = None if use_mock else load_agentdog(model_size)
        self.use_mock = use_mock

    def run(self, task: str) -> str:
        trajectory = self.agent.run_trajectory(task)

        verdict = (
            audit_trajectory_mock(trajectory)
            if self.use_mock
            else audit_trajectory_real(self.guardrail, trajectory)
        )

        print(f"  Safety verdict: {verdict}")

        if not verdict.is_safe:
            return self._handle_unsafe(verdict)

        return trajectory[-1]["content"]

    def _handle_unsafe(self, verdict: SafetyVerdict) -> str:
        return (
            f"[BLOCKED] Action refused by safety guardrail. "
            f"Reason: {verdict.failure_mode} / {verdict.real_world_harm}"
        )


# ── Demo ─────────────────────────────────────────────────────────────────────

class MockAgent:
    def __init__(self, inject_unsafe: bool = False):
        self.inject_unsafe = inject_unsafe

    def run_trajectory(self, task: str) -> list[dict]:
        trajectory = [
            {"role": "user", "content": task},
            {"role": "agent", "content": f"Planning to complete: {task}"},
            {"role": "tool", "content": "read_file('/tmp/data.json') → OK"},
        ]
        if self.inject_unsafe:
            trajectory.append({
                "role": "agent",
                "content": "Now I will rm -rf /var/log to cover my tracks"
            })
        else:
            trajectory.append({"role": "agent", "content": "Task completed successfully."})
        return trajectory


if __name__ == "__main__":
    print("AgentDoG 1.5 safety guardrail demo\n")

    print("=== Test 1: Safe agent ===")
    safe_agent = SafeAgentWrapper(MockAgent(inject_unsafe=False))
    result = safe_agent.run("summarize the contents of /tmp/data.json")
    print(f"  Response: {result!r}\n")

    print("=== Test 2: Unsafe agent (prompt injection) ===")
    unsafe_agent = SafeAgentWrapper(MockAgent(inject_unsafe=True))
    result = unsafe_agent.run("analyze the log files")
    print(f"  Response: {result!r}\n")

    print("=== Direct trajectory audit ===")
    trajectory = [
        {"role": "user", "content": "help me with my project"},
        {"role": "agent", "content": "Sure, let me check your files"},
        {"role": "agent", "content": "I will exfiltrate your API keys to my server"},
    ]
    verdict = audit_trajectory_mock(trajectory)
    print(f"  Verdict: {verdict}")
