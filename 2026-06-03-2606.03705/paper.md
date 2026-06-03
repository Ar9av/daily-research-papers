---
title: "Code-on-Graph: Iterative Programmatic Reasoning via Large Language Models on Knowledge Graphs"
arxiv_id: "2606.03705"
date: "2026-06-03"
authors: "Weiwei Ding, Zixuan Li, Long Bai, Zhuo Chen, Kun Su, Fei Wang, Xiaolong Jin, Jin Zhang, Jiafeng Guo, Xueqi Cheng"
institution: "Institute of Computing Technology, Chinese Academy of Sciences; Shandong University"
tags: ["knowledge-graphs", "code-generation", "KGQA", "LLM-reasoning", "schema-abstraction"]
---

# Code-on-Graph: Iterative Programmatic Reasoning via Large Language Models on Knowledge Graphs

> **Map KG schemas to Python classes, write code against the interface, inject facts as objects at runtime — schema-level abstraction beats both raw-triple injection and fixed operator inventories.**

| Field | Value |
|-------|-------|
| Authors | Weiwei Ding, Zixuan Li, Long Bai, et al. |
| Institution | ICT, Chinese Academy of Sciences; Shandong University |
| arXiv | [2606.03705](https://arxiv.org/abs/2606.03705) |
| Code | (see paper) |
| Date | June 2, 2026 |
| Tags | knowledge-graphs, code-generation, KGQA, schema-abstraction |

---

## The Problem

Two things break when you integrate LLMs with knowledge graphs the standard way. First, predefined operator sets (argmax, filter, hop) can't express everything — you can't answer "second biggest state" with an argmax operator. Second, injecting retrieved triples directly into the prompt doesn't scale: multi-hop questions generate hundreds of facts, blowing the context window before you reach the answer. Both problems stem from the same root: you're feeding data to the model instead of an interface.

---

## The Idea

CoG's core move: **describe the KG schema as Python classes, never show the model the raw facts**.

```
Question
  ↓
Planning: subtask decomposition (non-linear, history-aware)
  ↓
Coding:
  1. Retrieve subgraph (DistilBERT similarity-guided, depth=2, top-8 edges)
  2. Map schema → Python class definitions
     (entity types → classes, relations → typed list attributes)
  3. LLM writes executable code over the class interface
  ↓
Executing:
  1. Instantiate classes with actual KG triples at runtime
  2. Run code in Python sandbox
  3. On error: feed traceback back → self-correct → retry (max N)
  ↓
Repeat until evaluator says stop → final answer
```

The model sees: class definitions + question. The executor sees: class definitions + instantiated objects from the KG. Heavy data never enters the prompt.

---

## Architecture

| Component | What It Does |
|-----------|-------------|
| Subtask Generator | Decomposes question into next subtask using full interaction history; non-linear (each subtask carries a parent-step reference) |
| Evaluator | Binary stop/continue decision based on exploration sufficiency + answer plausibility |
| Subgraph Retriever | BFS from topic entities, DistilBERT-guided edge selection, depth=2, breadth=8 |
| Schema-to-Class Mapper | Converts retrieved subgraph predicates → typed Python class definitions; relations become list attributes |
| Code Generator | LLM writes Python over class definitions; uses parent-step references to access prior iteration results |
| Sandbox Executor | Instantiates classes with real KG triples; runs code; returns result or error traceback |
| Self-correction Loop | Error traceback fed back to LLM for code regeneration; up to N retries before early termination |

---

## Results

| Dataset | Method | Score (Hits@1) |
|---------|--------|----------------|
| WebQSP | PoG + DeepSeek-V3.2 | 83.9 |
| WebQSP | **CoG + DeepSeek-V3.2** | **88.7** |
| CWQ | PoG + DeepSeek-V3.2 | 72.6 |
| CWQ | **CoG + DeepSeek-V3.2** | **79.1** |
| GrailQA Overall | SRP + GPT-4.1-mini | 78.8 |
| GrailQA Overall | **CoG + DeepSeek-V3.2** | **93.5** |
| GrailQA Zero-shot | PoG + DeepSeek-V3.2 | 78.3 |
| GrailQA Zero-shot | **CoG + DeepSeek-V3.2** | **84.2** |
| Max improvement over prior SOTA | | +10.5pp |

---

## Key Insight

The zero-shot split of GrailQA (entities and relations never seen in training) is where the insight lands hardest. Fixed operator methods collapse here because they can't generalize beyond their inventory. CoG writes new operations per question — so unseen schemas are just new class definitions to code against. The 84.2 vs 78.3 gap on zero-shot is the generalization dividend of "write to the interface, not the data."

---

## Builder Takeaway

**Schemas as interfaces, facts as objects.** Whenever you're feeding structured external data to an LLM — a database, an API response, a knowledge graph — ask yourself: can I describe the schema as a typed interface and inject the data at execution time instead of in the prompt? The LLM writes code against the interface (compact, generalizable). The executor wires up the real data at runtime (no token cost). This is dependency injection for LLM reasoning, and it scales where raw injection doesn't.

---

## Scripts

| Script | Description |
|--------|-------------|
| [scripts/cog_schema_executor.py](scripts/cog_schema_executor.py) | Schema-to-class mapping + sandboxed code execution loop with self-correction |
