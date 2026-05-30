---
name: agent-radar-attention-steering
description: Apply AGENT-RADAR's spatial+temporal+semantic attention steering to any multi-agent pipeline to prevent context dilution as conversation histories grow long.
trigger: building a multi-agent system with multiple rounds of communication (debate, critique loops, planning agents) where performance degrades as history grows
---

# AGENT-RADAR Attention Steering Pattern

## When to use

- Your multi-agent system runs multiple rounds (debate, planning, critique, refinement loops)
- You're seeing hallucinations or reasoning drift as conversations get longer
- Agents are losing track of the original task constraints buried in long histories
- You're running AutoGen, GPTSwarm, LangGraph, or any custom multi-agent framework
- You want a training-free, plug-in improvement — no fine-tuning, no architecture changes

## Pattern

```
1. SCORE    — For each message in history, compute sentence-level relevance scores
2. DECAY    — Weight by spatial decay (graph hop distance) × temporal decay (message age)
3. RETRIEVE — Select sentences above threshold θ using combined score × semantic similarity
4. STEER    — Inject selected sentences as attention anchors before agent inference
5. PRESERVE — Never delete the full transcript — only steer, never prune
```

## Implementation sketch

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

encoder = SentenceTransformer("all-MiniLM-L6-v2")

def agent_radar_select(
    history: list[dict],       # [{"agent_id", "round", "sentences": [...], "hop_distance": int}]
    current_query: str,
    current_round: int,
    lambda_s: float = 0.5,     # spatial decay rate
    lambda_t: float = 0.7,     # temporal decay rate
    theta: float = 0.3,        # relevance threshold
) -> list[str]:
    """Returns sentences to steer attention toward."""
    query_emb = encoder.encode([current_query])
    selected = []

    for msg in history:
        hop = msg["hop_distance"]
        age = current_round - msg["round"] - 1

        # Spatial and temporal decay
        spatial = lambda_s ** max(0, hop - 1)   # direct neighbors = 1.0
        temporal = lambda_t ** max(0, age)

        # Sentence-level semantic scoring
        sentences = msg["sentences"]
        sent_embs = encoder.encode(sentences)
        sims = cosine_similarity(query_emb, sent_embs)[0]

        for sentence, sim in zip(sentences, sims):
            score = spatial * temporal * sim
            if score >= theta:
                selected.append(sentence)

    # Always include the current query
    return [current_query] + selected


def build_steered_prompt(base_prompt: str, selected_context: list[str]) -> str:
    """Prepend selected sentences as an explicit attention anchor."""
    anchor = "\n".join(f"[RELEVANT CONTEXT] {s}" for s in selected_context)
    return f"{anchor}\n\n{base_prompt}"


# Drop-in for any multi-agent round
class AgentRadarWrapper:
    def __init__(self, agent, agent_graph, lambda_s=0.5, lambda_t=0.7, theta=0.3):
        self.agent = agent
        self.graph = agent_graph  # networkx graph of agent topology
        self.lambda_s = lambda_s
        self.lambda_t = lambda_t
        self.theta = theta

    def run_round(self, agent_id: str, query: str, full_history: list[dict], round_num: int) -> str:
        # Compute hop distances from agent_id to all history message sources
        history_with_hops = [
            {**msg, "hop_distance": self._hop(agent_id, msg["agent_id"])}
            for msg in full_history
        ]
        selected = agent_radar_select(
            history_with_hops, query, round_num,
            self.lambda_s, self.lambda_t, self.theta
        )
        steered_prompt = build_steered_prompt(query, selected)
        return self.agent.generate(steered_prompt, full_history)  # full history preserved

    def _hop(self, source: str, target: str) -> int:
        import networkx as nx
        try:
            return nx.shortest_path_length(self.graph, source, target)
        except nx.NetworkXNoPath:
            return 99  # unreachable → near-zero weight
```

## Tuning the decay parameters

| Parameter | Low value | High value | Start with |
|---|---|---|---|
| `lambda_s` (spatial) | Heavily prefer nearby agents | Distant agents contribute equally | 0.5 |
| `lambda_t` (temporal) | Only recent messages matter | Old messages equally weighted | 0.7 |
| `theta` (threshold) | More context selected | Less context, higher precision | 0.3 |

For dense communication graphs (fully connected), raise `lambda_s` toward 0.8 — spatial differentiation matters less when everyone talks to everyone.

## Framework integrations

**AutoGen:**
```python
# Wrap each agent's reply function
original_reply = agent.generate_reply
agent.generate_reply = lambda messages, sender: agent_radar_wrapper.run_round(
    agent.name, messages[-1]["content"], flatten_history(messages), round_num
)
```

**LangGraph:**
```python
# Add as a node in the graph that pre-processes context before agent nodes
def radar_node(state):
    selected = agent_radar_select(state["history"], state["query"], state["round"])
    state["steered_context"] = selected
    return state
```

## Pitfalls

| What existing systems do | What AGENT-RADAR does instead |
|---|---|
| Compress/summarize history → lose subtle signals | Steer attention while preserving full transcript |
| Prune agents/edges → structural information loss | Score all messages, just weight them differently |
| Flat retrieval ignoring who said what, when | Decay by hop distance AND message age |
| Whole-message retrieval → noise from irrelevant sentences | Sentence-level segmentation → only the useful parts |
| Performance degrades with more agents/rounds | Robust scaling — decay factors absorb growing history |

## Key numbers

- +7.41 average absolute improvement over vanilla MAS baseline (5 benchmarks)
- +7.64 over AgentDropout (best prior method) on HotpotQA
- +12.87 F1 on MuSiQue when plugged into GPTSwarm
- +8.20 on MATH-500 competition math
- Works across random, layered, and fully-connected topologies
- Training-free: no weight updates, compatible with any LLM backbone

## Source

arxiv: [2605.30136](https://arxiv.org/abs/2605.30136) — AGENT-RADAR, Purdue University, May 2026
