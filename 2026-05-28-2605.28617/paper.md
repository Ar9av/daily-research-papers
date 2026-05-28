---
title: "LACUNA: Safe Agents as Recursive Program Holes"
arxiv_id: "2605.28617"
date: 2026-05-28
authors: ["Yaoyu Zhao", "Yichen Xu", "Oliver Bračevac", "Cao Nguyen Pham", "Frank Zhengqing Wu", "Martin Odersky"]
institution: "EPFL, Lausanne, Switzerland"
tags: [agents, safety, types, code-as-action, scala, llm, programming-models]
---

# LACUNA: Safe Agents as Recursive Program Holes

## Problem

LLM agents that write code (code-as-action) have a hard split between the runtime — which owns the loop, context, and control flow — and the model-written code, which only fills in individual actions. Letting model code shape the runtime would make agents more expressive, but existing safety defenses (sandboxes, policy languages, input hardening) are piecemeal: **none of them checks a whole generated action before it starts**. A partial execution that fails mid-way leaves the environment in an inconsistent state, and a prompt injection that reaches the runtime reaches further than one that only shapes a single action.

## Solution

LACUNA treats each agent action as a **typed hole** in the surrounding program. The core primitive is:

```scala
def agent[T](task: String): T
```

When execution reaches this call, the LLM writes Scala code intended to produce a value of type `T`. LACUNA compiles that code **against the live lexical scope** before any of it runs. If it type-checks: it executes. If it fails: the environment is unchanged, and the compiler's error messages are fed back as retry context. The same check also bounds which tools, files, and network handles the generated snippet may access (capture checking).

## Architecture

| Component | What it does |
|-----------|-------------|
| **Typed hole `agent[T]`** | Core primitive — LLM fills in code for a statically declared result type |
| **Pre-execution type check** | Compiles generated snippet against live lexical scope before any execution |
| **Capture checking** | Bounds which tools/files/network the snippet may use — enforced by Scala 3's type system |
| **Compiler-driven retry** | Rejected snippets leave env unchanged; diagnostics are fed back as model input |
| **Recursive agent calls** | Generated code can call `agent[T]` again — sub-agents, parallel decomposition, multi-model planning as ordinary control flow |
| **Scala 3 REPL session** | Cross-turn state persistence for multi-turn conversations |

## Results

**Type-system correctness:**
- ~400 hand-crafted test cases (well-formed + ill-formed snippets): **all pass**

**BrowseComp-Plus** (hard information-seeking, deepseek-v4-flash):
- **27.1%** accuracy (judge-scored)
- **8.6%** of generations rejected pre-execution — caught before they could run
- **0.7** retries per query on average; two-thirds of queries need zero retries
- **91.4%** end-to-end compile-success rate
- 5.9 rounds, 15.5 searches per query

**τ²-bench** (multi-turn conversation + tool use, 392 tasks, 4 domains):
- LACUNA (deepseek): **76.0%** overall vs Tool Calling baseline 90.0%
- LACUNA (gemini-lite): **29.1%** vs Tool Calling 33.2%
- Competitive with native tool-calling agents while adding static safety guarantees

## Key insight

The type checker is the safety layer — and it's almost free. 8.6% of LLM generations were structurally wrong and got caught before touching the environment, at a cost of only 0.7 retries/query. Two-thirds of queries never needed a retry. You get pre-execution rejection of bad actions for near-zero overhead, as a byproduct of the type system you'd already have if you built in a typed language.

## Builder takeaway

If you're building code-as-action agents, you're one type annotation away from pre-execution safety. The compiler's error message is free retry signal — you don't need a separate error handling system, a sandbox, or a policy engine. Build the agent call into the type system and let the compiler catch bad generations before they run. The host language's static guarantees extend to runtime-generated code.
