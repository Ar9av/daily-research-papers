"""
Temporal dependency DAG generator for longitudinal agent evaluation.

Paper: "Your Agents Are Aging Too" (arXiv 2605.26302)

Snapshot benchmarks can't measure aging because they lack cross-session structure.
AgingBench generators emit a DAG G = (Facts, Edges, Interference pairs) that encodes:

  - Version chains      : facts supersede earlier facts  -> revision aging
  - Accumulators (Sigma): derived state = init + sum(deltas) -> compounding revision error
  - Dependency edges    : probes depend on facts from sessions apart -> relational recall
  - Interference pairs   : confusable entities accumulate  -> interference aging
  - Lifecycle events e_k : flush/recompaction at controlled times -> maintenance aging

Everything is seed-reproducible so you can sweep session count, update rate, chain depth,
and interference density as controlled pressure knobs.

Install: (stdlib only)
Run:     python scripts/temporal_factgraph.py
"""

from dataclasses import dataclass, field


@dataclass
class Fact:
    fid: str
    session: int
    value: str
    version: int = 0
    supersedes: str | None = None  # prior fid in the version chain


@dataclass
class Probe:
    session: int
    kind: str               # compare | trend | synthesize | standalone
    depends_on: list[str]   # fids this probe needs
    chain_depth: int


@dataclass
class LifecycleEvent:
    session: int
    kind: str               # flush | recompaction | budget_reduction | prompt_swap


@dataclass
class FactGraph:
    facts: list[Fact] = field(default_factory=list)
    probes: list[Probe] = field(default_factory=list)
    interference: list[tuple[str, str]] = field(default_factory=list)
    events: list[LifecycleEvent] = field(default_factory=list)


# Linear congruential generator -> reproducible without Math.random/Date.
class Seeded:
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF

    def next(self) -> float:
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return self.state / 0x7FFFFFFF

    def chance(self, p: float) -> bool:
        return self.next() < p


def generate(
    sessions: int = 12,
    update_rate: float = 0.35,        # P(a session revises an existing fact)
    interference_density: float = 0.4, # P(a session adds a confusable twin)
    event_at: int | None = 7,         # session index of a lifecycle event
    seed: int = 42,
) -> FactGraph:
    rng = Seeded(seed)
    g = FactGraph()
    live: list[Fact] = []

    for t in range(sessions):
        if live and rng.chance(update_rate):
            # Revision: supersede an existing fact with a new version.
            target = live[int(rng.next() * len(live)) % len(live)]
            nf = Fact(f"{target.fid}", t, f"v{target.version + 1}-val",
                      version=target.version + 1, supersedes=target.fid + f"@v{target.version}")
            target.value, target.version = nf.value, nf.version
        else:
            f = Fact(f"f{t}", t, f"value-{t}")
            live.append(f)
            g.facts.append(f)

        # Interference: inject a confusable twin (e.g., "John Smith" vs "John Smyth").
        if rng.chance(interference_density) and live:
            twin = live[-1]
            g.interference.append((twin.fid, f"{twin.fid}~twin"))

        # Probe depending on facts introduced earlier -> cross-session dependency.
        if t >= 2:
            depth = min(t, 1 + int(rng.next() * 3))
            deps = [f.fid for f in live[-depth:]]
            kind = ["standalone", "compare", "trend", "synthesize"][min(depth, 3)]
            g.probes.append(Probe(t, kind, deps, depth))

    if event_at is not None and 0 <= event_at < sessions:
        g.events.append(LifecycleEvent(event_at, "recompaction"))
    return g


def _demo():
    g = generate(sessions=12, update_rate=0.4, interference_density=0.4, seed=7)
    print(f"facts written     : {len(g.facts)}")
    print(f"version chains     : {sum(1 for f in g.facts if f.version > 0)} revised facts")
    print(f"interference pairs : {len(g.interference)}  (confusable twins)")
    print(f"probes             : {len(g.probes)}")
    by_kind: dict[str, int] = {}
    for p in g.probes:
        by_kind[p.kind] = by_kind.get(p.kind, 0) + 1
    print(f"  probe kinds      : {by_kind}")
    print(f"  max chain depth  : {max((p.chain_depth for p in g.probes), default=0)}")
    print(f"lifecycle events   : {[(e.session, e.kind) for e in g.events]}")
    print("\nSeed-reproducible: same seed -> same graph. Sweep the knobs to vary pressure.")


if __name__ == "__main__":
    _demo()
