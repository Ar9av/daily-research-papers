"""
AGENT-RADAR — Spatial + temporal + semantic attention steering
arxiv: 2605.30136

Requires: sentence-transformers, numpy
Install: pip install sentence-transformers numpy

Run:
    python scripts/agent_radar.py
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field


@dataclass
class AgentMessage:
    agent_id: str
    round: int
    content: str
    sentences: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.sentences:
            # Simple sentence split — production: use nltk or spacy
            self.sentences = [s.strip() for s in self.content.split(".") if s.strip()]


def spatial_decay(hop_distance: int, lambda_s: float = 0.5) -> float:
    """Exponential decay over graph hop distance. Direct neighbors = 1.0."""
    return lambda_s ** max(0, hop_distance - 1)


def temporal_decay(age: int, lambda_t: float = 0.7) -> float:
    """Exponential decay over message age (rounds). Most recent = 1.0."""
    return lambda_t ** max(0, age)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x ** 2 for x in a))
    norm_b = math.sqrt(sum(x ** 2 for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class AgentRadar:
    """
    AGENT-RADAR: training-free attention steering for multi-agent systems.
    Scores every sentence in the history and selects the most relevant
    as attention anchors for the current agent's inference step.
    """

    def __init__(
        self,
        lambda_s: float = 0.5,   # spatial decay rate
        lambda_t: float = 0.7,   # temporal decay rate
        theta: float = 0.25,     # relevance threshold
        use_embeddings: bool = False,  # True: use sentence-transformers; False: TF-IDF mock
    ):
        self.lambda_s = lambda_s
        self.lambda_t = lambda_t
        self.theta = theta
        self.use_embeddings = use_embeddings
        self._encoder = None

        if use_embeddings:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if self._encoder:
            return self._encoder.encode(texts).tolist()
        # Mock: TF-IDF bag-of-words for demonstration
        vocab = {}
        for text in texts:
            for w in text.lower().split():
                if w not in vocab:
                    vocab[w] = len(vocab)
        def bow(text):
            vec = [0.0] * len(vocab)
            for w in text.lower().split():
                if w in vocab:
                    vec[vocab[w]] += 1.0
            return vec
        return [bow(t) for t in texts]

    def select_context(
        self,
        history: list[AgentMessage],
        current_query: str,
        current_round: int,
        hop_distances: dict[str, int],  # agent_id → hop distance from current agent
    ) -> list[str]:
        """
        Returns sentences from history to steer attention toward.
        Always includes current_query.
        """
        all_sentences = []
        sentence_scores = []

        for msg in history:
            hop = hop_distances.get(msg.agent_id, 99)
            age = current_round - msg.round - 1

            sp = spatial_decay(hop, self.lambda_s)
            tp = temporal_decay(age, self.lambda_t)
            spatio_temporal = sp * tp

            all_sentences.extend(msg.sentences)
            sentence_scores.extend([spatio_temporal] * len(msg.sentences))

        if not all_sentences:
            return [current_query]

        # Encode all sentences + query together
        texts = all_sentences + [current_query]
        embeddings = self._encode(texts)
        query_emb = embeddings[-1]
        sent_embs = embeddings[:-1]

        selected = [current_query]
        for sentence, st_score, sent_emb in zip(all_sentences, sentence_scores, sent_embs):
            sem_score = cosine_similarity(sent_emb, query_emb)
            final_score = st_score * sem_score
            if final_score >= self.theta:
                selected.append(sentence)

        return selected

    def build_steered_prompt(self, base_query: str, selected: list[str]) -> str:
        """Build the attention-steered prompt for the agent."""
        if len(selected) <= 1:  # only the query itself
            return base_query
        context_lines = "\n".join(
            f"[CONTEXT] {s}" for s in selected if s != base_query
        )
        return f"{context_lines}\n\n{base_query}"


# ── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("AGENT-RADAR attention steering demo\n")

    radar = AgentRadar(lambda_s=0.5, lambda_t=0.7, theta=0.15)

    # Simulate a multi-agent debate history
    history = [
        AgentMessage("planner", round=1, content=(
            "The task is to analyze Q3 sales data. "
            "We should focus on revenue trends. "
            "Start by collecting the raw numbers."
        )),
        AgentMessage("worker", round=2, content=(
            "I collected the data. Revenue is up 12% YoY. "
            "The main driver was enterprise contracts. "
            "Unrelated note: the weather is nice today."
        )),
        AgentMessage("critic", round=3, content=(
            "The analysis looks correct. "
            "However, we should also look at churn rate. "
            "Revenue alone doesn't tell the full story."
        )),
        AgentMessage("planner", round=4, content=(
            "Good point. Let us also check customer retention. "
            "The original task was Q3 revenue analysis. "
            "Please provide a final summary."
        )),
    ]

    current_query = "What are the key findings from the Q3 revenue analysis?"
    current_round = 5

    # Hop distances from the "critic" agent (current agent)
    hop_distances = {
        "planner": 1,   # direct neighbor
        "worker": 1,    # direct neighbor
        "critic": 0,    # self
    }

    print("History messages:")
    for msg in history:
        print(f"  [{msg.agent_id}, round {msg.round}] {msg.content[:60]}...")

    print(f"\nCurrent query: {current_query!r}")
    print(f"Current round: {current_round}")

    selected = radar.select_context(history, current_query, current_round, hop_distances)

    print(f"\nSelected context ({len(selected)} sentences):")
    for s in selected:
        print(f"  • {s!r}")

    print("\nSteered prompt:")
    print(radar.build_steered_prompt(current_query, selected))
