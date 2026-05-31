# LACUNA — Safe Agents via Typed Holes

> **Type-check agent-generated code before it runs. Catch unsafe actions at compile time, not runtime.**

| | |
|---|---|
| **Paper** | LACUNA: Safe Agents as Recursive Program Holes |
| **Authors** | Yaoyu Zhao, Yichen Xu, Oliver Bračevac, Cao Nguyen Pham, Frank Zhengqing Wu, Martin Odersky |
| **Institution** | EPFL (Martin Odersky's group — creator of Scala) |
| **arxiv** | [2605.28617](https://arxiv.org/abs/2605.28617) |
| **Date** | May 2026 |
| **Tags** | agents, type-safety, program-synthesis, formal-verification, tool-use |

---

## The Problem

LLM agents generate code and tool calls that run directly in the environment. There's no compile step, no type check, no contract verification — just "generate → execute → hope." Unsafe tool calls (wrong argument types, missing permissions, out-of-bounds access) only fail at runtime, potentially after irreversible side effects have already happened.

---

## The Idea

Represent every agent action as a **typed hole** — a placeholder in a well-typed program that the LLM must fill. Before execution, the filled program is type-checked. If it fails the type check, the agent must revise. Safe actions are guaranteed by the type system before any side effect occurs.

```
Agent generates: ⟨ hole ⟩ : FileWrite(path: Path, content: String) → IO[Unit]
LLM fills hole:  writeFile("/etc/passwd", "malicious")
Type checker:    ✗  /etc/passwd not in allowed Path set
                 → reject before execution, ask agent to revise
```

---

## Architecture

| Component | What it does |
|---|---|
| **Typed Hole** | A placeholder with a full type signature the LLM must satisfy |
| **Type Checker** | Verifies filled holes against the type signature before execution |
| **Recursive Holes** | Complex tasks decompose into sub-holes, each independently type-checked |
| **Safety Contracts** | Domain-specific types encode permissions (e.g., `AllowedPath`, `ReadOnlyFS`) |
| **Revision Loop** | On type error, the agent receives the error message and retries — no human needed |

---

## Results

- Eliminates entire classes of unsafe actions that pass LLM-based safety filters
- Type errors caught pre-execution: zero unsafe side effects from type-check failures
- Recursive hole decomposition handles multi-step tasks without loss of safety guarantees
- Compatible with any LLM that can generate code — no fine-tuning required

---

## Key Insight

Types are a better safety filter than prompting. "Don't do anything unsafe" is unverifiable. "This argument must be of type `AllowedPath`" is mechanically checkable. LACUNA shifts safety from a prompt-level concern to a compile-time guarantee — the same insight that makes typed languages safer than untyped ones.

---

## Builder Takeaway

If your agent generates code or structured tool calls, wrap the output in typed holes before execution. Define domain-specific types for your permission model (`ReadOnlyPath`, `ApprovedURL`, `BoundedInt`). Any generated value that violates a type gets rejected with a structured error the agent can fix — no runtime crash, no side effect.

---

## Scripts

| Script | What it shows |
|---|---|
| [`scripts/typed_holes.py`](scripts/typed_holes.py) | Typed hole definition, fill, type-check, and revision loop |
