---
name: openskill-evolution
description: Bootstrap a skill-learning loop from a task prompt and open-world resources — no curated skills, trajectories, or ground-truth verifier. Acquires verification anchors from documentation, synthesizes a self-built test suite, and iteratively refines skills supervision-free.
trigger: When building self-improving coding agents, tool-use agents, or any agent that needs skills without benchmark supervision; when you want to extract test cases from documentation rather than human annotation; when skills must transfer across different LLMs without model-specific tuning
---

## When to use

- Building coding or tool-use agents that need to self-improve on new task domains without labeled data
- You have documentation/repos but no ground-truth tests — need to synthesize a verification signal
- Skills must transfer to multiple models without per-model rewriting
- Evaluating agent self-improvement without risking GT data leakage into the training loop
- Any scenario where "read the docs, build the tests, improve against them" is the right loop

## Pattern

1. **Stage 1 — Knowledge Acquisition**: query docs, repos, tutorials for (a) task knowledge `ki` and (b) verification anchors `kv` — independently checkable facts: known dataset stats, expected API outputs, documented format constraints
2. **Leakage barrier**: filter all open-world queries to exclude benchmark identifiers; GT tests must not enter until Stage 3
3. **Synthesize skill plan** from `(task, env, ki)` — architecture, key procedures, domain rules
4. **Virtual Verifier**: separate LLM session generates deterministic pytest suite from `kv` alone — no GT access
5. **Evolution loop** (≤J=3 rounds):
   - Execute skill → run virtual tests → compute proxy pass rate
   - On failure: classify failure as bug vs. knowledge gap
   - Bug → revise skill from diagnostic; Gap → targeted retrieval → inject new knowledge → revise
   - Terminate when all virtual tests pass or budget exhausted
6. **Stage 3**: freeze final skill artifact; deploy zero-shot to any target agent; evaluate against GT tests here only

## Implementation

```python
class OpenSkillPipeline:
    def __init__(self, open_world: OpenWorldRetriever, verifier_llm, agent_llm, J=3):
        self.retriever = open_world
        self.verifier_llm = verifier_llm
        self.agent_llm = agent_llm
        self.J = J

    def run(self, task: str, env: Environment) -> Skill:
        # Stage 1: knowledge + verification anchors
        ki = self.retriever.retrieve_knowledge(task, env)      # docs, APIs, best practices
        kv = self.retriever.retrieve_anchors(task, env)        # checkable facts only
        plan = self.agent_llm.synthesize_plan(task, env, ki)

        # Build virtual test suite — no GT access
        virtual_tests = self.verifier_llm.generate_tests(task, env, kv)  # deterministic pytest

        # Stage 2: leakage-free evolution
        skill = self.agent_llm.generate_skill(plan, ki)
        for j in range(self.J):
            result = env.execute(skill)
            proxy_pass_rate = virtual_tests.run(result)
            if proxy_pass_rate == 1.0:
                break
            diagnostic = virtual_tests.diagnose(result)  # per-assertion + root cause
            if diagnostic.is_knowledge_gap:
                k_gap = self.retriever.retrieve_targeted(diagnostic.gap_query)
                skill = self.agent_llm.refine(skill, diagnostic, ki + k_gap)
            else:
                skill = self.agent_llm.refine(skill, diagnostic, ki)

        return skill  # portable artifact — deploy to any model

    def evaluate(self, skill: Skill, gt_tests, env: Environment) -> float:
        # Stage 3: GT unlocked here only
        result = env.execute_with_skill(skill)
        return gt_tests.run(result)
```

See `scripts/openskill.py` for a full runnable implementation with mock components.

## Tuning

| Parameter | Default | Notes |
|-----------|---------|-------|
| Refinement rounds J | 3 | Performance peaks at 3; overfitting to virtual tests observed at 5–10 |
| Skill count per task | 1–4 (from plan) | Larger sets cover more sub-problems; diminishing returns above 4 |
| Verification anchor types | Stats, ranges, formats, domain standards | More anchor types → better virtual test coverage |
| Leakage filter | Exclude benchmark name + identifiers | Critical — any GT leakage invalidates the supervision-free claim |
| Transfer target | Any model, no adaptation | Skills are text artifacts; no model-specific rewriting needed |

## Pitfalls

| Old Approach | This Paper |
|-------------|------------|
| Skills from parametric LLM knowledge | Skills from live open-world retrieval — more current, domain-specific |
| Verify against GT tests during training | Build virtual tests from docs; GT seen only at final eval |
| Skills coupled to one model | Portable skill artifacts transfer to Haiku, Qwen, DeepSeek, Mistral as-is |
| Human-written test suites | Virtual verifier covers 88.9% of GT intents + adds 3.4× more assertions |
| Iterate until GT passes | Iterate until virtual tests pass; cap at J=3 to avoid overfitting |
| Knowledge gap = prompt again | Classify bug vs. gap; targeted retrieval for gaps specifically |

## Key numbers

- SkillsBench overall: 43.6% (Opus) / 42.1% (GPT) vs. human 44.5% / 44.8%
- +8.9pp / +8.8pp over strongest closed-world baseline
- Virtual verifier covers 88.9% of GT test intents, zero GT access
- Virtual verifier generates 3.4× more test functions, 15.3 additional assertions/task
- Cross-model transfer: +5.5pp to +14.8pp over no-skill on 4 weaker models
- AutoSkill skills collapse on other models; OpenSkill skills don't

## Source

- arXiv: https://arxiv.org/abs/2606.06741
- Code: https://github.com/OpenLAIR/OpenSkill
