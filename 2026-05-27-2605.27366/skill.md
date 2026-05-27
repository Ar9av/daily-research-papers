---
name: muse-skill-lifecycle
description: Apply the MUSE skill lifecycle when building agents that reuse capabilities across tasks — create on-demand, unit-test before storing, accumulate per-skill memory, transfer across agents.
trigger: building an AI agent that executes repeating sub-tasks, needs to persist reusable capabilities, or should improve over time without retraining
---

# MUSE Skill Lifecycle Pattern

## When to use

- You are building an agent that solves similar sub-tasks repeatedly (code generation, data extraction, API calls)
- You want agent capabilities to compound — get better the more it runs — without fine-tuning
- You need skills that are portable across different agent runtimes or models
- You are hitting context-window limits on long-horizon tasks and need structured state persistence

## Pattern

```
1. RETRIEVE  — Before executing a sub-task, query the skill store for a matching skill
2. CREATE    — If no skill exists, execute normally; extract the successful approach as a skill
3. VALIDATE  — Run unit tests on the new skill before storing it; discard if tests fail
4. STORE     — Persist the skill with metadata: tags, trigger condition, creation context
5. RECORD    — After every invocation (success or failure), append to the skill's experience log
6. REFINE    — On test failure or low success rate, re-generate the skill using logged failures as context
7. TRANSFER  — Skills are plain text/code — export and inject into other agents unchanged
```

The key loop: every task run either reuses an existing skill (and logs the outcome) or produces a new one. The skill store grows; success rates improve; no task starts cold.

## Implementation sketch

```python
class SkillStore:
    def retrieve(self, task_description: str) -> Skill | None:
        # semantic search over stored skills by trigger condition
        ...

    def store(self, skill: Skill) -> bool:
        # run unit tests first; return False if any fail
        if not self.validate(skill):
            return False
        self.db[skill.name] = skill
        return True

    def record(self, skill_name: str, outcome: Outcome):
        # append to per-skill experience log
        self.db[skill_name].history.append(outcome)

    def refine(self, skill_name: str) -> Skill:
        # re-generate using failure history as context
        failures = [h for h in self.db[skill_name].history if not h.success]
        return self.llm.generate_skill(context=failures)


class Agent:
    def execute(self, task: Task) -> Result:
        skill = self.skill_store.retrieve(task.description)

        if skill:
            result = skill.run(task)
            self.skill_store.record(skill.name, Outcome(task, result))
            return result

        # no skill found — execute raw, then extract
        result = self.llm.run(task)
        if result.success:
            new_skill = self.llm.extract_skill(task, result)
            self.skill_store.store(new_skill)  # validates before storing

        return result
```

## Skill memory schema

```python
@dataclass
class Skill:
    name: str
    trigger: str           # when to invoke (natural language)
    implementation: str    # code or instructions
    unit_tests: list[str]  # test cases that must pass before storage
    history: list[Outcome] # every invocation result — the per-skill memory

@dataclass
class Outcome:
    task: Task
    result: Result
    success: bool
    notes: str             # failure reason or adaptation made
```

## Pitfalls (from the paper's gap analysis)

| What most systems do | What MUSE does instead |
|----------------------|----------------------|
| Create skills outside the runtime loop | Create skills from within execution — full access to live context |
| One shared memory for everything | Separate skill-level memory per skill — tracks individual skill history |
| Store skills without testing | Unit-test every skill before storage — no garbage in the library |
| Flat conversation history | Adaptive compression + cross-session state — no context overflow |
| Skills tied to one agent | Plain-text skills that transfer to any agent unchanged |

## Source

arxiv: [2605.27366](https://arxiv.org/abs/2605.27366) — MUSE-Autoskill, ByteDance + RIT, May 2026
