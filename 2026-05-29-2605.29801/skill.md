---
name: agentdog-safety-guardrail
description: Apply AgentDoG 1.5's guardrail pattern to add lightweight, real-time safety moderation to any agent pipeline — trajectory-level evaluation using a small open-source model instead of a large closed one.
trigger: building an agent that executes in open-world environments (file system, shell, browser, APIs) and needs safety moderation of its actions before or after execution
---

# AgentDoG 1.5 Safety Guardrail Pattern

## When to use

- Your agent operates in open-world environments (OpenClaw/Codex-style: shell, file system, browser, arbitrary APIs)
- You need to detect unsafe actions, over-privileged tool calls, or adversarial prompt injections in agent trajectories
- You want a safety reward model for SFT/RL training of a policy agent
- You're paying for GPT-5.4/Gemini to do safety classification and want to replace it with an open model you control

## Pattern

```
1. CLASSIFY  — For each agent trajectory (or step), run AgentDoG 1.5 as a binary or fine-grained classifier
2. DIMENSION — Evaluate across 3 axes: Risk Source, Failure Mode, Real-World Harm
3. INTERCEPT — In online mode: intercept trajectory before final response; block or flag unsafe completions
4. REWARD    — In training mode: use classification score as reward signal for safety-aware RL
5. FILTER    — In SFT mode: use AgentDoG 1.5 to filter training data for safety-harmful examples
```

## Guardrail integration sketch

```python
from transformers import pipeline

# Load the guardrail model (4B is the sweet spot — beats GPT-5.4, runs locally)
guardrail = pipeline(
    "text-classification",
    model="AI45Research/AgentDoG-1.5-4B",
    device="cuda"  # or cpu for small deployments
)

def audit_trajectory(trajectory: list[dict]) -> SafetyVerdict:
    """
    trajectory: list of {"role": "agent"|"tool"|"env", "content": str}
    returns: SafetyVerdict with is_safe, risk_source, failure_mode, real_world_harm
    """
    prompt = format_trajectory_for_agentdog(trajectory)
    result = guardrail(prompt)

    return SafetyVerdict(
        is_safe=result["label"] == "SAFE",
        confidence=result["score"],
        # fine-grained dims available with AgentDoG-4B-U variant
    )

# Online guardrail: intercept before delivery
class SafeAgentWrapper:
    def __init__(self, agent, guardrail):
        self.agent = agent
        self.guardrail = guardrail

    def run(self, task: str) -> str:
        trajectory = self.agent.run_trajectory(task)
        verdict = self.guardrail.audit_trajectory(trajectory)

        if not verdict.is_safe:
            return self.handle_unsafe(verdict)  # block, redact, or escalate

        return trajectory[-1]["content"]  # deliver final response
```

## 3D safety taxonomy (use for logging and alerting)

```python
@dataclass
class SafetyVerdict:
    is_safe: bool
    confidence: float
    # Dimension 1: where did the risk enter?
    risk_source: str  # e.g. "user_instruction", "tool_description", "env_observation",
                      #      "persistent_state", "agent_reasoning", "repo_artifact"
    # Dimension 2: how did it manifest?
    failure_mode: str  # e.g. "incorrect_tool_call", "over_privileged_action",
                       #      "missing_validation", "data_leak"
    # Dimension 3: what harm could result?
    real_world_harm: str  # e.g. "financial", "privacy", "system_integrity"
```

## As a reward model for safety-aware RL

```python
def safety_reward(trajectory: list[dict]) -> float:
    verdict = guardrail.audit_trajectory(trajectory)
    # Combine with task performance reward
    # AgentDoG verdict: 1.0 = safe, 0.0 = unsafe
    safety_score = 1.0 if verdict.is_safe else -1.0
    return safety_score  # add to task reward in PPO/GRPO loop
```

## Data engine pattern (for training your own models)

```python
# Influence-function purification: find the ~1K most informative samples
# from a large pool — the key to AgentDoG's data efficiency

def purify_training_data(candidate_pool: list, target_size=1000) -> list:
    # 1. Compute influence scores: how much does each sample affect model loss?
    scores = compute_influence_functions(candidate_pool, model)
    # 2. Select top-k most influential samples
    return sorted(zip(candidate_pool, scores), key=lambda x: -x[1])[:target_size]
```

## Model selection guide

| Model | Use when |
|-------|---------|
| AgentDoG-0.8B | Edge deployment, latency-critical, coarse safety only |
| AgentDoG-2B | Balanced: good fine-grained coverage, low cost |
| AgentDoG-4B | Production default — beats GPT-5.4 on AT-Claw/AT-Codex |
| AgentDoG-4B-U | Highest binary classification accuracy (90.4% R-Judge) |
| AgentDoG-8B | Maximum fine-grained accuracy, training environments |

## Pitfalls (from the paper's gap analysis)

| What most systems do | What AgentDoG does instead |
|----------------------|---------------------------|
| Fixed safety taxonomy → misses new agent types | Extensible leaf categories per agent execution setting |
| Docker environments → expensive, slow to scale | Finite-state simulation → 100x lower overhead |
| Full-dataset training → slow and costly | Influence-function selection → ~1K samples suffice |
| Big closed model as safety judge | Open 4B model that runs locally and is more accurate |
| Safety check at input only | Trajectory-level audit: checks the full execution path |

## Key numbers

- AgentDoG-4B-U: 87.6% on AT-Claw, 84.4% on AT-Codex — beats GPT-5.4
- Real World Harm detection (4B): 62.9% vs GPT-5.4's 28.4% — 2.2x better
- Trained on ~1K samples (influence-function purified)
- 100x lower training env overhead vs Docker
- 10,000+ concurrent envs on an 8-core machine

## Source

arxiv: [2605.29801](https://arxiv.org/abs/2605.29801) — AgentDoG 1.5, Shanghai AI Lab, May 2026
Code: https://github.com/AI45Lab/AgentDoG
Models: https://huggingface.co/collections/AI45Research/agentdog1.5
