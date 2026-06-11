---
name: in-context-video-conditioning
description: Bypass explicit pose/skeleton intermediates in video generation by concatenating the driving video directly into the diffusion context sequence. Use mode-specific shifted RoPE to disambiguate multiple input streams without training conflicts. Apply Bias-Aware DPO post-training to fix fine-grained detail failures caused by synthetic data bias.
trigger: When building video generation or character animation systems using a DiT/I2V backbone; when pose skeleton extraction causes information loss (complex interactions, animals, occluded joints); when you want to unify multiple animation sub-tasks in one model; when post-training quality on fine-grained motions (hands, fingers) is insufficient.
---

## When to use

- You're building on a DiT-based I2V backbone (Wan, CogVideoX, etc.) and need motion conditioning
- Pose skeleton extraction fails on your target content (animals, complex interactions, overlapping characters)
- You want one model to handle animation + replacement + multi-character scenarios
- You need cross-identity motion transfer (driving character ≠ reference character)
- Post-training fine-grained detail (hands, fingers) needs improvement without full retraining

## Pattern

1. **Context concatenation**: encode reference image and driving video with the same VAE; concatenate token sequences as `[zref ; zt ; zdriv]`. Add fixed spatial offset ΔW to driving stream to prevent token collision.
2. **28-channel mask latent**: segment reference and driving frames into K+1 binary channels (1 env-switch + K binding slots, K=6). Downsample 8× spatially, stack 4 frames temporally → 28 channels. Concatenate to context.
3. **Mode-specific shifted RoPE**: assign distinct coordinate spaces to each stream — animation mode uses temporal shift (zref t=0, zt t=1..Tv), replacement mode uses spatial shift (zref h+=ΔH). This prevents the model from confusing mode behaviors.
4. **Reverse driving training**: generate synthetic driving ỹ from real video y, train with ỹ as input and y as supervised target. Never let synthetic artifacts enter the supervision signal.
5. **Post-training**: construct Bias-Aware DPO pairs (r=1 error round, r⁻=2 error rounds) for failure regions (hands). Run 400 DPO steps.

## Implementation

```python
# Mask compression: RGB mask video → 28-channel binary latent
def extract_mask_latent(mask_frames_THWC, K=6, spatial_factor=8, temporal_stride=4):
    T, H, W, _ = mask_frames_THWC.shape
    N_ch = K + 1  # 7 channels: 1 env + K binding slots

    # 1. Classify each pixel into one of 7 color classes → (T, N_ch, H, W)
    binary = rgb_to_binary_channels(mask_frames_THWC)  # (T, 7, H, W)

    # 2. Spatial downsample to match VAE latent grid: (T, 7, H/8, W/8)
    binary_ds = avg_pool_spatial(binary, factor=spatial_factor)

    # 3. Temporal stacking: 4 frames → 1 latent frame, stacked along channels
    # (T, 7, Hl, Wl) → (T_lat, 28, Hl, Wl)
    T_lat = T // temporal_stride
    latent = binary_ds.reshape(T_lat, temporal_stride, N_ch, Hl, Wl)
    latent = latent.transpose(0, 2, 1, 3, 4).reshape(T_lat, N_ch * temporal_stride, Hl, Wl)
    return latent.transpose(1, 0, 2, 3)  # (28, T_lat, Hl, Wl)


# Mode-specific RoPE coordinate assignment (Table 1 from paper)
def assign_rope_coords(Tv, Hv, Wv, delta_W, delta_H, mode):
    if mode == "animation":
        ref_t   = [0]               # frozen temporal reference
        video_t = range(1, Tv+1)    # temporal separation
        driv_t  = range(1, Tv+1)
        ref_h   = range(Hv)
        ref_w   = range(Wv)
    else:  # replacement
        ref_t   = [0]               # no temporal separation...
        video_t = range(0, Tv)
        driv_t  = range(0, Tv)
        ref_h   = range(delta_H, delta_H + Hv)  # ...spatial separation instead
        ref_w   = range(Wv)
    driv_w = range(delta_W, delta_W + Wv)       # always spatially detached
    return {"ref": (ref_t, ref_h, ref_w),
            "video": (video_t, range(Hv), range(Wv)),
            "driving": (driv_t, range(Hv), driv_w)}


# Bias-Aware DPO preference pair construction
def build_dpo_pair(y, P, G, R, S):
    p_y = P(y)           # extract pose from real video
    s   = G(p_y, S)      # driving video with reference S
    r   = G(p_y, R)      # positive: 1 round of error
    p_r = P(r)           # re-extract from r (accumulate error)
    r_neg = G(p_r, R)    # negative: 2 rounds of error
    R1 = r.frames[randint(len(r.frames))]
    return (s, R1, r, r_neg)   # (driving, ref_frame, positive, negative)
```

See `scripts/scail2_components.py` for full runnable implementation tested on st3ve.

## Tuning

| Parameter | Default | Notes |
|-----------|---------|-------|
| K (binding slots) | 6 | Channels Ch1-ChK; up to 6 characters per frame; Ch0=env switch |
| Total mask channels | 28 | = 4(K+1) = 4×7; must match temporal stride |
| Temporal stride | 4 | Match VAE temporal compression; adjust if VAE differs |
| ΔW (driving spatial offset) | 16 latent cols | Keeps driving tokens spatially detached; avoid collision |
| ΔH (ref spatial offset, replacement) | 10 latent rows | Must not overlap video stream h range |
| DPO training steps | 400 | On top of 3,500 main training steps |
| DPO error rounds | 2 (negative) vs 1 (positive) | Wider gap: use weaker P'' for 2nd extraction pass |

## Pitfalls

| Old Approach | This Paper |
|-------------|------------|
| Pose skeleton as intermediate | Direct driving video in context — no information loss |
| Separate models per sub-task | Unified model with mode-specific conditioning |
| Inject driving via channel concat | Concatenate to token sequence → model uses full attention over all streams |
| Train on synthetic targets | Reverse driving: real video as supervision, synthetic as input |
| SFT for hand detail | Bias-Aware DPO: explicit negative pairs with accumulated pose error |
| Single RoPE grid for all streams | Mode-specific shifts prevent temporal/spatial ambiguity between animation and replacement |

## Key numbers

- FVD 287.11 (vs SCAIL 309.63, Wan-Animate 305.31) on Studio-Bench
- SSIM 0.6453, PSNR 19.09, LPIPS 0.2231
- Motion Consistency: 68.3% win vs SCAIL, 65.0% win vs Kling 3.0
- Multi-character Identity Isolation: 93.3% win vs MultiAnimate (zero-shot)
- Character replacement: 67.9% environment integration win vs MoCha
- 14B param backbone (Wan2.1), 64×H100, ~1 week training
- MotionPair-60K: animation:replacement ≈ 3:1, <30% discard rate

## Source

- arXiv: https://arxiv.org/abs/2606.10804
- Code: https://github.com/zai-org/SCAIL-2
- Project page: https://teal024.github.io/SCAIL-2/
