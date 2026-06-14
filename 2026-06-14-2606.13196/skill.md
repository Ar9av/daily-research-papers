---
name: designics-creativity-checklist
description: Evaluate any AI system against the 10 Designics requirements for genuine machine creativity. Identifies whether a system is truly creative (participates in recursive intervention dynamics) or primarily generative (produces creative-looking outputs without understanding why problems exist or what changed).
trigger: When auditing an AI system's creative capability beyond output quality; when designing agentic systems and deciding which requirements to prioritize; when evaluating whether ethics is internal to your system's reasoning or added as a post-generation filter.
---

## When to use

- You're building an AI agent and want to assess where it sits on the creativity spectrum
- You're deciding which architectural additions would most advance genuine creative capability
- You want to distinguish "generative" from "creative" when scoping AI system claims
- You're auditing whether ethical constraints are internal to your system or bolted on after generation

## The 10 Requirements (by Law)

### Law of Perception
| Req | What to check |
|-----|--------------|
| R1 | Does the system maintain a model of the environment it acts in? |
| R2 | Can it revise what it pays attention to based on what it learned? |
| R5 | Does it observe what its actions actually changed? |
| R7 ★ | Does it rescope when consequences reveal a wrong frame? |
| R9 ★ | Are ethical/social constraints part of perception, not post-generation filters? |

### Law of Conflict
| Req | What to check |
|-----|--------------|
| R3 ★ | Does it identify what needs fixing without being told? |

### Law of Capability
| Req | What to check |
|-----|--------------|
| R4 | Does it enact changes, not just produce artifacts for human review? |
| R6 | Does it update its knowledge from observed consequences? |
| R8 | Do local decisions compound into emergent global structure? |
| R10 | Does it actively support human agency, not just optimize task performance? |

★ = critical gaps in most current AI systems

## Where Current Systems Land

| System type | Strong | Critical gaps |
|-------------|--------|---------------|
| Foundation model (LLM) | R4 | R3, R5, R7, R9, R10 |
| Agentic tool loop | R4, partial R5/R6 | R3, R7, R9 |
| Scientific discovery (FunSearch, AlphaFold) | R4, R8 | R3, R7, R9 |
| Human-AI teaming with workload reallocation | R5, partial R10 | R3, R7 |

## Pattern

1. **Score each requirement**: full / partial / weak / none (see script)
2. **Identify critical gaps**: R3, R7, R9 are unsatisfied in most systems
3. **Target the loop-closing gaps first**: R5 (consequence observation) → R6 (update) → R7 (rescope) form the most actionable cluster for agentic systems
4. **Address R3 structurally**: anomaly detection, environment monitoring, and proactive problem-finding move an agent toward conflict identification
5. **Address R9 architecturally**: ethical constraints must enter at the perception/scoping layer, not only at the output layer

## Architectural moves that advance specific requirements

| Gap | What helps |
|-----|-----------|
| R3 (no conflict identification) | Structured environment monitoring; anomaly detection over state; proactive health-check tools |
| R5 (no consequence observation) | Explicit state-diff after each action; environment snapshots before/after |
| R7 (no rescoping) | Scope-revision hooks triggered by unexpected consequences; meta-reasoning about what to attend to next |
| R9 (ethics as filter) | Constitutional constraints at the system prompt / perception layer, not only output safety filters |
| R10 (human agency) | Human-in-the-loop checkpoints; measuring human workload and adjusting task allocation |

## Key numbers

- Critical requirements unsatisfied by most current AI: R3, R7, R9
- Typical RAG-backed agent score: ~32% (3.2/10) under this framework
- The paper identifies 3 laws and 10 requirements derived from recursive intervention dynamics
- Verified computationally in 2 domains: cyber-physical (geometric mesh generation) and cyber-biological (human-AI teaming with EEG workload tracking)

## Source

- arXiv: https://arxiv.org/abs/2606.13196
