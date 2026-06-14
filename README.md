# daily-research-papers

One AI paper per day — plain-English summaries, coding-agent skills, and runnable Python scripts.

Each folder is `YYYY-MM-DD-<arxiv-id>/` and contains:
- `paper.md` — what the paper says, key numbers, builder takeaway
- `skill.md` — how to apply the paper's ideas in a coding agent
- `scripts/` — runnable Python implementations of the core ideas

---

## Index

| Date | Paper | Skill | Scripts |
|------|-------|-------|---------|
| 2026-06-14 | [Creative Machine](2026-06-14-2606.13196/paper.md) — 10 requirements for genuine machine creativity (Designics); R3/R7/R9 unsatisfied by all current AI; ethics must be internal to perception, not a post-generation filter | [skill](2026-06-14-2606.13196/skill.md) | [scripts/](2026-06-14-2606.13196/scripts/) |
| 2026-06-13 | [CODE](2026-06-13-2601.13112/paper.md) — one poisoned RAG document causes 5–25× reasoning token inflation with no accuracy drop; tri-agent attack: Contradiction Architect + Conflict Weaver + Style Adapter | [skill](2026-06-13-2601.13112/skill.md) | [scripts/](2026-06-13-2601.13112/scripts/) |
| 2026-06-12 | [SpatialClaw](2026-06-12-2606.13673/paper.md) — persistent Python kernel as action interface for spatial VLM agents; +11.2 pp over prior SOTA, 20 benchmarks, no model-specific tuning | [skill](2026-06-12-2606.13673/skill.md) | [scripts/](2026-06-12-2606.13673/scripts/) |
| 2026-06-11 | [SCAIL-2](2026-06-11-2606.10804/paper.md) — end-to-end character animation without pose skeletons; 28-channel mask conditioning + mode-specific RoPE; 93.3% win on multi-character identity isolation | [skill](2026-06-11-2606.10804/skill.md) | [scripts/](2026-06-11-2606.10804/scripts/) |
| 2026-06-10 | [Natural Backdoors in CodeLMs](2026-06-10-2606.10846/paper.md) — clean-trained code models hide triggers from data bias; one token swap moves an insecure snippet to #1 in search; only unlearning fixes it | [skill](2026-06-10-2606.10846/skill.md) | [scripts/](2026-06-10-2606.10846/scripts/) |
| 2026-06-09 | [V-JEPA 2](2026-06-09-2506.09985/paper.md) — latent world model from internet video + 62hr robot fine-tuning; 75% pick-and-place zero-shot, 15× faster planning than video diffusion | [skill](2026-06-09-2506.09985/skill.md) | [scripts/](2026-06-09-2506.09985/scripts/) |
| 2026-06-08 | [OpenSkill](2026-06-08-2606.06741/paper.md) — supervision-free agent self-evolution: builds skills + test suite from docs alone, 88.9% GT coverage, within 1–3pp of human | [skill](2026-06-08-2606.06741/skill.md) | [scripts/](2026-06-08-2606.06741/scripts/) |
| 2026-06-07 | [SpatialUncertain](2026-06-07-2605.30557/paper.md) — VLMs score below random at knowing when perspective misleads them; visual input makes it worse (-35pp) | [skill](2026-06-07-2605.30557/skill.md) | [scripts/](2026-06-07-2605.30557/scripts/) |
| 2026-06-06 | [Knowledge Infusion Layers](2026-06-06-2606.06356/paper.md) — 4-layer framework for where knowledge enters generative models; 70.97% toxicity reduction, frozen backbones | [skill](2026-06-06-2606.06356/skill.md) | [scripts/](2026-06-06-2606.06356/scripts/) |
| 2026-06-05 | [Benchmark Agent](2026-06-05-2606.06462/paper.md) — autonomous benchmark construction: 20× faster than humans, 97% acceptance, +17pp over direct LLM generation | [skill](2026-06-05-2606.06462/skill.md) | [scripts/](2026-06-05-2606.06462/scripts/) |
| 2026-06-04 | [AgingBench](2026-06-04-2605.26302/paper.md) — agents age after deployment even with frozen weights; four mechanisms (compression/interference/revision/maintenance), stage-level repair — 85% recall drop, 4.5× half-life from policy alone | [skill](2026-06-04-2605.26302/skill.md) | [scripts/](2026-06-04-2605.26302/scripts/) |
| 2026-06-03 | [Code-on-Graph](2026-06-03-2606.03705/paper.md) — LLM reasons over KGs by writing code against schema-class interfaces, not raw triples — +10.5pp over SOTA, 93.5 on GrailQA | [skill](2026-06-03-2606.03705/skill.md) | [scripts/](2026-06-03-2606.03705/scripts/) |
| 2026-06-02 | [HypoAgent](2026-06-02-2605.31370/paper.md) — Surgical hypothesis repair over knowledge graphs: diagnose which fragment broke, fix only that — 78.1→94.0 Jaccard | [skill](2026-06-02-2605.31370/skill.md) | [scripts/](2026-06-02-2605.31370/scripts/) |
| 2026-06-01 | [AutoSci](2026-06-01-2605.31468/paper.md) — Full research lifecycle agent: typed memory + Trust Guard + self-evolution → 6.3/10 ICLR score in 27 hours | [skill](2026-06-01-2605.31468/skill.md) | [scripts/](2026-06-01-2605.31468/scripts/) |
| 2026-05-31 | [Saguaro (SSD)](2026-05-31-2603.03251/paper.md) — 30% faster than speculative decoding, 5× over autoregressive — parallelizes draft and verify | [skill](2026-05-31-2603.03251/skill.md) | [scripts/](2026-05-31-2603.03251/scripts/) |
| 2026-05-30 | [AGENT-RADAR](2026-05-30-2605.30136/paper.md) — Training-free attention steering for multi-agent systems: +7.4 pts avg across 5 benchmarks | [skill](2026-05-30-2605.30136/skill.md) | [scripts/](2026-05-30-2605.30136/scripts/) |
| 2026-05-29 | [AgentDoG 1.5](2026-05-29-2605.29801/paper.md) — 4B open model beats GPT-5.4 on agent safety, trained on ~1K samples | [skill](2026-05-29-2605.29801/skill.md) | [scripts/](2026-05-29-2605.29801/scripts/) |
| 2026-05-28 | [LACUNA](2026-05-28-2605.28617/paper.md) — Safe agents via typed holes: type-check generated code before it runs | [skill](2026-05-28-2605.28617/skill.md) | [scripts/](2026-05-28-2605.28617/scripts/) |
| 2026-05-27 | [MUSE-Autoskill](2026-05-27-2605.27366/paper.md) — Self-evolving agents via skill creation, memory & evaluation | [skill](2026-05-27-2605.27366/skill.md) | [scripts/](2026-05-27-2605.27366/scripts/) |
