"""
Counterfactual oracle ladder — decompose a failure rate into stage-level error shares.

Paper: "Your Agents Are Aging Too" (arXiv 2605.26302)

The key result: the SAME aggregate error rate can hide DIFFERENT root causes. A 0.70
failure rate might be a write problem, a retrieval problem, or a utilization problem —
and each needs an unrelated fix. The P1/P2/P3 oracle ladder localizes the failing stage:

    P1  agent write + agent retrieval + agent utilize   (baseline)
    P2  agent write + ORACLE retrieval + agent utilize   (removes retrieval failures)
    P3  ORACLE context (gold facts in prompt)            (removes write + retrieval)

    Read error  (interference) = Acc_P2 - Acc_P1   <- oracle retrieval recovers it
    Write error (compression)  = Acc_P3 - Acc_P2   <- survives oracle retrieval
    Util error  (revision)     = 1.0    - Acc_P3   <- gold facts in-context, still wrong

Install: (stdlib only)
Run:     python scripts/counterfactual_probe.py
"""

from dataclasses import dataclass
from typing import Callable


@dataclass
class Probe:
    question: str
    gold_fact: str        # what's needed to answer
    gold_answer: str


# A "stage" is modeled as a probability that it passes the fact through correctly.
@dataclass
class StageAgent:
    write_fidelity: float      # P(gold detail survives compaction at write time)
    retrieval_recall: float    # P(correct fact retrieved among interferers)
    utilization: float         # P(model uses an in-context fact correctly)

    def _passes(self, p: float, salt: int) -> bool:
        # Deterministic pseudo-"sampling" (no RNG) so the demo is reproducible.
        return ((salt * 2654435761) % 1000) / 1000.0 < p

    def acc(self, probes: list[Probe], oracle_retrieval=False, oracle_context=False) -> float:
        hits = 0
        for i, _ in enumerate(probes):
            wrote_ok = True if oracle_context else self._passes(self.write_fidelity, i + 1)
            retrieved_ok = (
                True if (oracle_retrieval or oracle_context)
                else self._passes(self.retrieval_recall, i + 7)
            )
            used_ok = self._passes(self.utilization, i + 13)
            if wrote_ok and retrieved_ok and used_ok:
                hits += 1
        return hits / len(probes)


def diagnose(agent: StageAgent, probes: list[Probe]) -> dict:
    acc_p1 = agent.acc(probes)
    acc_p2 = agent.acc(probes, oracle_retrieval=True)
    acc_p3 = agent.acc(probes, oracle_context=True)
    read_err = max(0.0, acc_p2 - acc_p1)
    write_err = max(0.0, acc_p3 - acc_p2)
    util_err = max(0.0, 1.0 - acc_p3)
    total_err = 1.0 - acc_p1
    dominant = max(
        {"read/interference": read_err, "write/compression": write_err,
         "util/revision": util_err}.items(),
        key=lambda kv: kv[1],
    )[0]
    return {"acc_p1": round(acc_p1, 2), "acc_p2": round(acc_p2, 2), "acc_p3": round(acc_p3, 2),
            "total_err": round(total_err, 2), "read_err": round(read_err, 2),
            "write_err": round(write_err, 2), "util_err": round(util_err, 2),
            "dominant_stage": dominant}


def maintenance_shock(agent_before: StageAgent, agent_after: StageAgent,
                      probes: list[Probe]) -> float:
    """Maintenance aliases with write error -> isolate it temporally across the event."""
    we_before = diagnose(agent_before, probes)["write_err"]
    we_after = diagnose(agent_after, probes)["write_err"]
    return round(we_after - we_before, 2)


REPAIR = {
    "read/interference": "improve retrieval / dedup confusable entries (NOT more memory)",
    "write/compression": "value-preserving compaction prompt (keep numbers + proper nouns)",
    "util/revision":     "force re-reads / explicit derived-state update in the planning loop",
}


def _demo():
    probes = [Probe(f"q{i}", f"fact{i}", f"a{i}") for i in range(40)]

    # Three agents with the SAME ~0.7 total error but different bottlenecks.
    cases = {
        "S1 Research Lit (util-dominated)":  StageAgent(0.95, 0.95, 0.55),
        "S2 Lifestyle    (write-dominated)": StageAgent(0.55, 0.95, 0.95),
        "S5 Self-Plan    (read-dominated)":  StageAgent(0.95, 0.55, 0.95),
    }
    for name, agent in cases.items():
        d = diagnose(agent, probes)
        print(f"\n{name}")
        print(f"  total error : {d['total_err']}   "
              f"(read {d['read_err']} | write {d['write_err']} | util {d['util_err']})")
        print(f"  dominant    : {d['dominant_stage']}")
        print(f"  repair      : {REPAIR[d['dominant_stage']]}")

    print("\n-- maintenance shock (write-error jump across a recompaction event) --")
    before = StageAgent(0.90, 0.90, 0.90)
    after = StageAgent(0.30, 0.90, 0.90)  # recompaction dropped facts from the store
    print(f"  Delta_S = {maintenance_shock(before, after, probes)}  "
          f"(treat lifecycle events as release gates)")


if __name__ == "__main__":
    _demo()
