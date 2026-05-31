# Saguaro — Speculative Speculative Decoding

> **30% faster than speculative decoding. 5x faster than autoregressive. Zero accuracy loss.**

| | |
|---|---|
| **Paper** | Speculative Speculative Decoding |
| **Authors** | Tanishq Kumar (Stanford), Tri Dao (Princeton / Together AI), Avner May (Together AI) |
| **arxiv** | [2603.03251](https://arxiv.org/abs/2603.03251) |
| **Code** | [github.com/tanishqkumar/ssd](https://github.com/tanishqkumar/ssd) |
| **Date** | May 2026 |
| **Tags** | inference, speculative-decoding, parallelism, LLM-serving |

---

## The Problem

Standard speculative decoding (SD) has a hidden sequential bottleneck:

```
Round 1:  [SPECULATE] → [VERIFY] → idle
Round 2:                            [SPECULATE] → [VERIFY] → idle
```

The draft model sits **idle during verification**. The verifier sits **idle during drafting**. Both GPUs are underutilized half the time.

---

## The Idea

Run speculation and verification **in parallel on separate hardware**. While the verifier is checking tokens from round T, the draft model predicts likely outcomes of that verification and **pre-speculates the next round for each**.

```
Round T:   Draft ──→ [PRE-SPECULATE all likely outcomes] ──────────────────→
           Target →→ [VERIFY round T-1] → sends outcome →→ cache hit? → done!
```

If the actual verification outcome matches a pre-computed speculation → **return immediately, zero draft overhead**. If not → fall back to regular SD (still lossless).

This is **Saguaro**, the optimized SSD algorithm.

---

## Architecture

| Component | What it does |
|---|---|
| **Speculation Cache** | Maps predicted verification outcomes → pre-computed token sequences |
| **Outcome Predictor** | Uses draft model logits to predict which tokens the verifier will accept + the bonus token. Up to 90% accuracy |
| **Balanced Sampler** | Navigates the tension between acceptance rate and prediction quality — new sampling algorithm |
| **Adaptive Fallback** | On cache miss, picks fallback strategy (primary vs backup speculator) based on batch size |
| **Primary Speculator** | Fast draft model on separate device (1×H100) |
| **Verifier** | Target model on TP=4 H100s |

---

## Results

**Llama-3.1-70B / 1B draft — TP=4 H100s, batch size 1:**

| Dataset | AR (tok/s) | SD (tok/s) | Saguaro (tok/s) | vs SD | vs AR |
|---|---|---|---|---|---|
| HumanEval | 54.7 | 176 | 283 | 1.60× | 5.17× |
| GSM8k | 54.7 | 188 | 301 | 1.60× | 5.50× |
| Alpaca | 54.7 | 145 | 224 | 1.55× | 4.10× |
| Ultrafeedback | 54.7 | 138 | 215 | 1.55× | 3.93× |
| **Average** | **54.7** | **161.8** | **255.8** | **1.58×** | **4.68×** |

**Qwen3-32B / 0.6B draft:**

| Average | AR | SD | Saguaro | vs SD | vs AR |
|---|---|---|---|---|---|
| | 88.8 | 136.8 | 203.8 | **1.49×** | **2.29×** |

- Saguaro is **~30% faster** than the strongest SD baseline on average
- **Outperforms SD even at large batch sizes** (+20%), unlike prior SSD methods
- Bonus token prediction accuracy: up to **90%**
- Compatible with EAGLE-3 and tree-based methods for further gains

---

## Key Insight

Verification outcome prediction is the crux. If you can predict what the verifier will accept — including the bonus token sampled from the residual distribution — you can pre-compute the right next speculation with high accuracy. Saguaro frames this as a constrained optimization problem and solves it using the draft model's own logit distribution.

---

## Builder Takeaway

If you're serving large language models and using speculative decoding today, Saguaro is a drop-in upgrade: move the draft model to a separate device, add a speculation cache, and start pre-computing outcomes in parallel. The fallback to regular SD means you never lose correctness. The 30%+ throughput gain is free.

---

## Scripts

| Script | What it shows |
|---|---|
| [`scripts/saguaro_cache.py`](scripts/saguaro_cache.py) | Speculation cache + async SSD loop skeleton |
