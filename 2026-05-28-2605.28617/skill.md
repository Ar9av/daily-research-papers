---
name: lacuna-typed-agent-holes
description: Apply LACUNA's typed-hole pattern when building code-as-action agents — type-check generated code before execution, use compiler errors as retry signals, bound tool access via the type system.
trigger: building a code-as-action agent where the LLM writes executable code, especially when safety, partial-execution failures, or tool access bounding matter
---

# LACUNA Typed-Hole Pattern

## When to use

- You are building a code-as-action agent (model writes code, not just tool calls)
- You need to prevent partial execution failures from leaving state inconsistent
- You want to bound which tools/resources model-generated code can access
- You need a retry loop without building a separate error handling system
- You want sub-agents and parallel decomposition without a separate orchestration layer

## Pattern

```
1. DECLARE  — At each action site, declare the expected result type T
2. GENERATE — LLM writes code intended to produce a value of type T
3. CHECK    — Type-check the generated code against the live scope BEFORE execution
4. BRANCH   — If check passes: execute. If fails: env is unchanged, send diagnostics to model
5. RETRY    — Model regenerates using compiler errors as feedback (converges in ~0.7 retries)
6. RECURSE  — Generated code may call agent[T] again — sub-agents are just nested typed holes
7. BOUND    — Capture checking enforces which tools/files/network the snippet may use
```

The key invariant: **a rejected snippet never executes**. No partial state changes, no rollback needed.

## Implementation sketch

```typescript
// Conceptual port — LACUNA is Scala 3, but the pattern applies anywhere
// with a compile/eval step and a type contract

interface AgentHole<T> {
  task: string;
  resultType: TypeDescriptor<T>;    // what the LLM must produce
  allowedCapabilities: Capability[]; // tools/files/network — capture checking
}

async function agent<T>(hole: AgentHole<T>): Promise<T> {
  const context = getLexicalScope(); // variables, tools available at call site

  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    const code = await llm.generate(hole.task, hole.resultType, context);

    const check = typeCheck(code, hole.resultType, context, hole.allowedCapabilities);

    if (check.ok) {
      return execute<T>(code, context); // only runs if well-typed
    }

    // env is unchanged — feed compiler errors back as retry context
    context.addFeedback(check.diagnostics);
  }

  throw new AgentError("Max retries exceeded");
}

// Recursive sub-agents: generated code calls agent() again — ordinary control flow
// Parallel: generated code calls Promise.all([agent(...), agent(...)]) — no special protocol
```

## Capability bounding

```typescript
// Declare what the agent hole is allowed to touch
const restrictedHole: AgentHole<SearchResult[]> = {
  task: "Search for recent papers on X",
  resultType: SearchResult.array(),
  allowedCapabilities: [
    Capability.webSearch,      // ✓ allowed
    // Capability.fileWrite    // ✗ not declared → type error if snippet tries to use it
    // Capability.shellExec    // ✗ not declared → rejected before execution
  ]
};
```

## Pitfalls (from the paper's gap analysis)

| What most code-as-action agents do | What LACUNA does instead |
|------------------------------------|-------------------------|
| Run generated code, catch errors at runtime | Type-check before execution — bad code never runs |
| Require rollback/undo on partial failure | No partial execution → no rollback needed |
| Piecemeal safety: sandbox + policy + input hardening | One unified check: does this code type-check against T in live scope? |
| Sub-agents require a separate orchestration protocol | Nested `agent[T]` calls — just ordinary recursive code |
| Tool access bounded by runtime policy evaluation | Bounded statically by capture checking at compile time |
| Retry needs a bespoke error classification system | Compiler diagnostics are the retry signal — free |

## Key numbers

- 8.6% of LLM generations rejected pre-execution (BrowseComp-Plus)
- 0.7 retries/query average; ⅔ of queries need zero retries
- 91.4% compile-success rate
- 76.0% task completion on τ²-bench (392 tasks)
- ~400 type-system test cases: all pass

## Source

arxiv: [2605.28617](https://arxiv.org/abs/2605.28617) — LACUNA, EPFL, May 2026
Co-authored by Martin Odersky (creator of Scala 3)
