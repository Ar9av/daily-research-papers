"""
SCAIL-2: Unifying Controlled Character Animation with End-to-end In-Context Conditioning
arxiv: 2606.10804  —  Yan, Guo, Yang, Tang; Tsinghua / Z.ai, 2026
GitHub: https://github.com/zai-org/SCAIL-2

Install: pip install numpy pillow
Run:     python scripts/scail2_components.py

This script implements the 3 core algorithmic contributions of SCAIL-2 that
don't require the 14B model weights, producing real numerical + visual results:

  1. In-Context Mask Compression   RGB mask frames → 28-channel binary latent
  2. Mode-Specific Shifted RoPE    3D coordinate grids for animation vs replacement
  3. Bias-Aware DPO                Error-propagation chain for preference dataset
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Literal

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ────────────────────────────────────────────────────────────────────────────
# 1. IN-CONTEXT MASK COMPRESSION
#    RGB mask frames → 28-channel binary latent
#    From: wan/utils/scail_utils.py :: extract_and_compress_mask_to_latent()
#
#    Color encoding (matches SCAIL-2 repo):
#      White  (255,255,255) → Ch0: environment visible (background keep)
#      Red    (255,  0,  0) → Ch1: character binding slot 1
#      Green  (  0,255,  0) → Ch2: character binding slot 2
#      Blue   (  0,  0,255) → Ch3: character binding slot 3
#      Yellow (255,255,  0) → Ch4: character binding slot 4
#      Magenta(255,  0,255) → Ch5: character binding slot 5
#      Cyan   (  0,255,255) → Ch6: character binding slot 6  (K=6 slots total)
#
#    Spatial:  H/8 × W/8  (same as VAE latent grid)
#    Temporal: 4 video frames stacked per latent frame  (VAE temporal stride)
#    Result:   7 channels × 4 temporal stride = 28 channels
# ────────────────────────────────────────────────────────────────────────────

MASK_COLORS = {
    "env":   np.array([255, 255, 255]),  # Ch0 — background/environment visible
    "char1": np.array([255,   0,   0]),  # Ch1
    "char2": np.array([  0, 255,   0]),  # Ch2
    "char3": np.array([  0,   0, 255]),  # Ch3
    "char4": np.array([255, 255,   0]),  # Ch4
    "char5": np.array([255,   0, 255]),  # Ch5
    "char6": np.array([  0, 255, 255]),  # Ch6
}
COLOR_ORDER = list(MASK_COLORS.keys())   # 7 channels: 1 env + 6 binding slots
N_MASK_CHANNELS = len(COLOR_ORDER)       # 7
TEMPORAL_STRIDE = 4                      # VAE temporal compression
TOTAL_LATENT_CHANNELS = N_MASK_CHANNELS * TEMPORAL_STRIDE  # 28


def rgb_frame_to_binary_channels(
    frame_hwc: np.ndarray,  # (H, W, 3) uint8 in [0, 255]
    tolerance: int = 20,
) -> np.ndarray:
    """
    Convert one RGB mask frame into 7 binary channel maps.
    Returns (7, H, W) float32 in {0, 1}.
    """
    H, W = frame_hwc.shape[:2]
    channels = np.zeros((N_MASK_CHANNELS, H, W), dtype=np.float32)
    for idx, (name, color) in enumerate(MASK_COLORS.items()):
        dist = np.abs(frame_hwc.astype(np.int32) - color).max(axis=-1)
        channels[idx] = (dist < tolerance).astype(np.float32)
    return channels


def spatial_downsample(channels_chw: np.ndarray, factor: int = 8) -> np.ndarray:
    """
    Average-pool spatial dimensions by `factor`.
    (C, H, W) → (C, H//factor, W//factor)
    """
    C, H, W = channels_chw.shape
    Hd, Wd = H // factor, W // factor
    out = channels_chw[:, :Hd*factor, :Wd*factor].reshape(C, Hd, factor, Wd, factor)
    return out.mean(axis=(2, 4))  # (C, Hd, Wd)


def extract_and_compress_mask_to_latent(
    mask_frames: np.ndarray,  # (T, H, W, 3) uint8 RGB mask video
    spatial_factor: int = 8,
) -> np.ndarray:
    """
    Full mask compression pipeline (matches SCAIL-2's scail_utils.py).

    Input:  (T, H, W, 3) RGB mask video
    Output: (28, T//TEMPORAL_STRIDE, H//8, W//8) float32 binary latent

    The 28 channels = 7 binary color channels × 4 temporal frames stacked.
    This encodes:
      - which spatial regions belong to each character (binding)
      - whether the environment source is reference or driving (environment switch)
    """
    T, H, W, _ = mask_frames.shape
    T_lat = T // TEMPORAL_STRIDE
    H_lat, W_lat = H // spatial_factor, W // spatial_factor

    # (T, 7, H, W)
    binary_all = np.stack([rgb_frame_to_binary_channels(mask_frames[t])
                           for t in range(T)], axis=0)

    # Downsample spatially: (T, 7, H_lat, W_lat)
    binary_ds = np.stack([spatial_downsample(binary_all[t], spatial_factor)
                          for t in range(T)], axis=0)

    # Temporal stride: stack 4 frames along channel dim
    # (T, 7, H_lat, W_lat) → (T_lat, 28, H_lat, W_lat)
    latent = binary_ds[:T_lat*TEMPORAL_STRIDE].reshape(
        T_lat, TEMPORAL_STRIDE, N_MASK_CHANNELS, H_lat, W_lat
    )  # (T_lat, 4, 7, H_lat, W_lat)
    latent = latent.transpose(0, 2, 1, 3, 4)   # (T_lat, 7, 4, H_lat, W_lat)
    latent = latent.reshape(T_lat, TOTAL_LATENT_CHANNELS, H_lat, W_lat)
    # Final shape expected by SCAIL2Model: (28, T_lat, H_lat, W_lat)
    return latent.transpose(1, 0, 2, 3)


def make_demo_mask_video(
    T: int = 8, H: int = 64, W: int = 64, n_chars: int = 2,
) -> np.ndarray:
    """
    Synthesise a simple demo mask video:
    - White background (environment channel)
    - Red character on left half, Green character on right half
    Returns (T, H, W, 3) uint8.
    """
    frames = []
    for t in range(T):
        frame = np.ones((H, W, 3), dtype=np.uint8) * 255  # white background
        # Red character: left half, animated (slightly drifts right over time)
        cx1 = W // 4 + t * 2
        x0, x1 = max(0, cx1 - W//8), min(W, cx1 + W//8)
        frame[H//4:3*H//4, x0:x1] = [255, 0, 0]
        if n_chars >= 2:
            # Green character: right half
            cx2 = 3*W//4
            x0, x1 = max(0, cx2 - W//8), min(W, cx2 + W//8)
            frame[H//4:3*H//4, x0:x1] = [0, 255, 0]
        frames.append(frame)
    return np.stack(frames)


# ────────────────────────────────────────────────────────────────────────────
# 2. MODE-SPECIFIC SHIFTED ROPE
#    3D RoPE coordinate grids for the 3 token streams (zref, zt, zdriv)
#    in animation mode vs replacement mode.
#
#    From Table 1 of the paper:
#
#    Animation Mode:
#      zref:  t=0,         h=[0,Hv),  w=[0,Wv)
#      zt:    t=[1,Tv],    h=[0,Hv),  w=[0,Wv)
#      zdriv: t=[1,Tv],    h=[0,Hv),  w=[ΔW, ΔW+Wv)
#
#    Replacement Mode:
#      zref:  t=0,         h=[ΔH,ΔH+Hv), w=[0,Wv)
#      zt:    t=[0,Tv-1],  h=[0,Hv),     w=[0,Wv)
#      zdriv: t=[0,Tv-1],  h=[0,Hv),     w=[ΔW, ΔW+Wv)
#
#    Key difference:
#    - Animation: zref and zt have temporal difference (0 vs 1..Tv)
#                 → model learns: ref is a frozen snapshot, not part of the timeline
#    - Replacement: zref has spatial shift ΔH
#                   → model learns: ref character lives in a different spatial region
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class RoPECoords:
    t: np.ndarray  # temporal coordinates  (T_lat,)
    h: np.ndarray  # height coordinates    (H_lat,)
    w: np.ndarray  # width coordinates     (W_lat,)
    stream: str    # "ref", "video", or "driving"
    mode: str      # "animation" or "replacement"


def build_rope_coords(
    Tv: int = 20, Hv: int = 8, Wv: int = 14,
    delta_W: int = 16, delta_H: int = 10,
    mode: Literal["animation", "replacement"] = "animation",
) -> dict[str, RoPECoords]:
    """
    Build the 3D RoPE coordinate grids for all 3 token streams.
    Matches Table 1 of the SCAIL-2 paper.

    Args:
        Tv:      number of latent video frames
        Hv, Wv:  spatial latent grid dimensions
        delta_W: fixed spatial offset for driving stream (keeps it detached)
        delta_H: spatial offset for reference in replacement mode
        mode:    "animation" or "replacement"

    Returns dict with keys "ref", "video", "driving", each a RoPECoords.
    """
    h_base = np.arange(Hv)
    w_base = np.arange(Wv)

    if mode == "animation":
        ref_t   = np.array([0])                    # single frozen timestep
        video_t = np.arange(1, Tv + 1)             # 1..Tv
        driv_t  = np.arange(1, Tv + 1)             # same as video

        ref   = RoPECoords(t=ref_t,   h=h_base,         w=w_base,          stream="ref",     mode=mode)
        video = RoPECoords(t=video_t, h=h_base,         w=w_base,          stream="video",   mode=mode)
        driv  = RoPECoords(t=driv_t,  h=h_base,         w=w_base+delta_W,  stream="driving", mode=mode)

    else:  # replacement
        ref_t   = np.array([0])                    # also t=0 (no temporal differentiation)
        video_t = np.arange(0, Tv)                 # 0..Tv-1
        driv_t  = np.arange(0, Tv)                 # same as video

        ref   = RoPECoords(t=ref_t,   h=h_base+delta_H, w=w_base,         stream="ref",     mode=mode)
        video = RoPECoords(t=video_t, h=h_base,          w=w_base,         stream="video",   mode=mode)
        driv  = RoPECoords(t=driv_t,  h=h_base,          w=w_base+delta_W, stream="driving", mode=mode)

    return {"ref": ref, "video": video, "driving": driv}


def rope_1d_encoding(coords: np.ndarray, dim: int = 16) -> np.ndarray:
    """
    Sinusoidal RoPE encoding for 1D coordinate sequence.
    Returns (len(coords), dim) — the rotation angles for each position.
    """
    freqs = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    angles = np.outer(coords, freqs)  # (N, dim//2)
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)  # (N, dim)


def token_overlap_analysis(
    anim_coords: dict[str, RoPECoords],
    repl_coords: dict[str, RoPECoords],
) -> dict:
    """
    Check that animation and replacement modes have no t-coordinate collision
    between the video and ref streams (the key design property of mode-specific RoPE).
    """
    results = {}
    for mode_name, coords in [("animation", anim_coords), ("replacement", repl_coords)]:
        ref_t = set(coords["ref"].t.tolist())
        vid_t = set(coords["video"].t.tolist())
        driv_t = set(coords["driving"].t.tolist())
        ref_h = set(coords["ref"].h.tolist())
        vid_h = set(coords["video"].h.tolist())

        if mode_name == "animation":
            # Key property: temporal separation between ref (t=0) and video (t=1..Tv)
            t_overlap = ref_t & vid_t
            spatial_h_overlap = ref_h & vid_h
            disambiguation = "TEMPORAL"
            is_disjoint = len(t_overlap) == 0
        else:
            # Key property: spatial separation via ΔH for ref
            t_overlap = ref_t & vid_t  # both t=0, so they overlap temporally!
            spatial_h_overlap = ref_h & vid_h
            disambiguation = "SPATIAL (ΔH)"
            is_disjoint = len(spatial_h_overlap) == 0

        results[mode_name] = {
            "disambiguation": disambiguation,
            "ref_t_range": (min(coords["ref"].t), max(coords["ref"].t)),
            "video_t_range": (min(coords["video"].t), max(coords["video"].t)),
            "ref_h_range": (min(coords["ref"].h), max(coords["ref"].h)),
            "video_h_range": (min(coords["video"].h), max(coords["video"].h)),
            "t_overlap": len(t_overlap),
            "spatial_h_overlap": len(spatial_h_overlap),
            "is_disjoint": is_disjoint,
        }
    return results


# ────────────────────────────────────────────────────────────────────────────
# 3. BIAS-AWARE DPO PREFERENCE CONSTRUCTION
#    Constructs (driving, reference, positive, negative) preference tuples
#    for post-training with DPO.
#
#    From Section 3.4 of the paper:
#      Given motion y, pose estimator P, generator G:
#        r  = G(P(y), R)          - positive: one round of P+G
#        r⁻ = G(P(r), R)          - negative: two rounds (extra error accumulated)
#
#    Preference tuple: (s=driving, R1=ref_frame, r=positive, r⁻=negative)
#    where s = G(P(y), S) with different reference S.
#
#    We simulate this with measurable "pose error" metrics to show that
#    r⁻ genuinely accumulates more error than r.
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class Pose:
    joints: np.ndarray   # (J, 2) joint positions
    confidence: float

    @property
    def is_hand_region(self) -> np.ndarray:
        return self.joints[-6:]   # last 6 joints = hands (simplified)


@dataclass
class SyntheticVideo:
    frames: np.ndarray   # (T, H, W, 3)
    source_pose: Pose
    generation_round: int = 0
    cumulative_error: float = 0.0


def mock_pose_estimator(video: SyntheticVideo, noise_scale: float = 0.03) -> Pose:
    """
    Simulates imperfect pose extraction.
    Each call adds noise proportional to `noise_scale`.
    Finger joints (last 6) get 3× more noise — matches paper's finding.
    """
    n_joints = 17
    true_joints = video.source_pose.joints.copy()
    noise = np.random.randn(n_joints, 2) * noise_scale
    noise[-6:] *= 3.0   # finger joints are harder to estimate
    noisy_joints = true_joints + noise
    conf = max(0.0, video.source_pose.confidence - noise_scale * 0.5)
    return Pose(joints=noisy_joints, confidence=conf)


def mock_animation_generator(
    pose: Pose,
    reference_video: SyntheticVideo,
    noise_scale: float = 0.02,
) -> SyntheticVideo:
    """
    Simulates G(pose, reference): adds rendering noise on top of pose error.
    """
    true_joints = reference_video.source_pose.joints
    # Rendered output drifts from ground truth by pose error + render noise
    render_noise = np.random.randn(*pose.joints.shape) * noise_scale
    rendered_joints = pose.joints + render_noise
    cumulative_err = float(np.linalg.norm(rendered_joints - true_joints))

    output_frames = reference_video.frames.copy()
    # Simulate visual effect of error by adding noise proportional to joint error
    err_magnitude = cumulative_err / len(true_joints)
    output_frames = np.clip(
        output_frames.astype(np.float32) + np.random.randn(*output_frames.shape) * err_magnitude * 50,
        0, 255
    ).astype(np.uint8)

    return SyntheticVideo(
        frames=output_frames,
        source_pose=Pose(joints=rendered_joints, confidence=pose.confidence * 0.95),
        generation_round=reference_video.generation_round + 1,
        cumulative_error=cumulative_err,
    )


def joint_error(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Mean joint position error, split by body vs hand joints."""
    n = len(pred)
    body_idx = slice(0, n - 6)
    hand_idx = slice(n - 6, n)
    return {
        "body_mpjpe": float(np.linalg.norm(pred[body_idx] - gt[body_idx], axis=-1).mean()),
        "hand_mpjpe": float(np.linalg.norm(pred[hand_idx] - gt[hand_idx], axis=-1).mean()),
        "total_mpjpe": float(np.linalg.norm(pred - gt, axis=-1).mean()),
    }


def construct_preference_dataset(
    n_samples: int = 10,
    seed: int = 42,
    use_weaker_estimator: bool = True,
) -> list[dict]:
    """
    Constructs DPO preference tuples (s, R1, r, r⁻).
    Measures that r⁻ consistently has higher pose error than r.

    Args:
        n_samples:             number of preference pairs to generate
        use_weaker_estimator:  if True, use less accurate P'' for r⁻
                               (Eq. 6 from paper, wider gap variant)
    """
    np.random.seed(seed)
    pairs = []

    for i in range(n_samples):
        # Ground-truth pose and source video
        gt_joints = np.random.randn(17, 2) * 0.5  # 17 joints, 2D
        gt_pose = Pose(joints=gt_joints, confidence=1.0)
        T, H, W = 8, 32, 32
        source_video = SyntheticVideo(
            frames=np.random.randint(0, 255, (T, H, W, 3), dtype=np.uint8),
            source_pose=gt_pose,
        )

        # Two different references R and S (different character appearances)
        ref_R = SyntheticVideo(frames=source_video.frames.copy(), source_pose=gt_pose)
        ref_S = SyntheticVideo(frames=np.random.randint(0, 255, (T, H, W, 3), dtype=np.uint8),
                               source_pose=gt_pose)

        # s = G(P(y), S)  — driving video with reference S
        pose_y = mock_pose_estimator(source_video, noise_scale=0.03)
        s = mock_animation_generator(pose_y, ref_S)

        # r = G(P(y), R)  — positive sample (1 round of error)
        r_pos = mock_animation_generator(pose_y, ref_R)

        # r⁻ = G(P(r), R) — negative sample (2 rounds of error accumulated)
        if use_weaker_estimator:
            pose_r = mock_pose_estimator(r_pos, noise_scale=0.06)   # weaker estimator P''
        else:
            pose_r = mock_pose_estimator(r_pos, noise_scale=0.03)   # same P
        r_neg = mock_animation_generator(pose_r, ref_R)

        # Reference frame R1: random frame from r
        r1_idx = np.random.randint(0, T)

        # Measure errors vs ground truth
        err_pos = joint_error(r_pos.source_pose.joints, gt_joints)
        err_neg = joint_error(r_neg.source_pose.joints, gt_joints)

        pairs.append({
            "sample_id": i,
            "driving": s,
            "ref_frame_idx": r1_idx,
            "positive": r_pos,
            "negative": r_neg,
            "pos_error": err_pos,
            "neg_error": err_neg,
            "hand_gap": err_neg["hand_mpjpe"] - err_pos["hand_mpjpe"],
            "body_gap": err_neg["body_mpjpe"] - err_pos["body_mpjpe"],
            "neg_worse": err_neg["total_mpjpe"] > err_pos["total_mpjpe"],
        })

    return pairs


# ────────────────────────────────────────────────────────────────────────────
# 4. VISUALISATION HELPERS
# ────────────────────────────────────────────────────────────────────────────

def visualize_mask_channels(
    mask_frames: np.ndarray,  # (T, H, W, 3)
    latent: np.ndarray,       # (28, T_lat, H_lat, W_lat)
    save_path: str | None = None,
) -> None:
    """
    Save a side-by-side grid:
      - Left column: first 4 original RGB mask frames (T=0..3)
      - Right columns: first 7 latent channels at T_lat=0 (the 7 binary decomposed channels)
    """
    if not HAS_PIL:
        print("  [visualize] PIL not available — skipping image save")
        return

    T_lat = latent.shape[1]
    # Show input frame 0 and 4 latent channel maps at t=0
    cell = 64
    n_show_channels = 7  # one per color
    total_w = cell * (1 + n_show_channels)
    total_h = cell

    canvas = Image.new("RGB", (total_w, total_h), (40, 40, 40))

    # Original frame (upscale to cell×cell)
    orig = Image.fromarray(mask_frames[0]).resize((cell, cell), Image.NEAREST)
    canvas.paste(orig, (0, 0))

    # Latent channels at t=0 (first T_lat frame covers frames 0..3 stacked)
    # latent shape: (28, T_lat, H_lat, W_lat)
    # channel i at latent t=0 corresponds to video frame i//7 * stride + (i%7)
    for ch in range(n_show_channels):
        ch_map = latent[ch, 0]  # (H_lat, W_lat)
        # Normalise to [0,255]
        img_arr = (ch_map * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr, mode="L").resize((cell, cell), Image.NEAREST)
        img_rgb = img.convert("RGB")
        # Tint to channel color for readability
        color = list(MASK_COLORS.values())[ch]
        tint = Image.new("RGB", (cell, cell), tuple(int(c) for c in color))
        blended = Image.blend(img_rgb, tint, alpha=0.3)
        canvas.paste(blended, ((ch + 1) * cell, 0))

    if save_path:
        canvas.save(save_path)
        print(f"  [visualize] Saved mask channel grid → {save_path}")
    else:
        canvas.save("/tmp/scail2_mask_channels.png")
        print("  [visualize] Saved mask channel grid → /tmp/scail2_mask_channels.png")


def print_rope_table(
    coords: dict[str, RoPECoords],
    mode: str,
) -> None:
    print(f"\n  {'─'*52}")
    print(f"  Mode-Specific RoPE Coordinates — {mode.upper()} MODE")
    print(f"  {'─'*52}")
    print(f"  {'Stream':<10} {'t range':<18} {'h range':<14} {'w range':<14}")
    print(f"  {'─'*52}")
    for name, c in coords.items():
        t_str = f"[{c.t.min()}, {c.t.max()}]"
        h_str = f"[{c.h.min()}, {c.h.max()}]"
        w_str = f"[{c.w.min()}, {c.w.max()}]"
        print(f"  {name:<10} {t_str:<18} {h_str:<14} {w_str:<14}")


# ────────────────────────────────────────────────────────────────────────────
# DEMO
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("SCAIL-2 Core Components Demo")
    print("arxiv: 2606.10804  |  github: zai-org/SCAIL-2")
    print("=" * 60)

    # ── 1. Mask Compression ────────────────────────────────────────────────
    print("\n[1] In-Context Mask Compression")
    print("    RGB mask video → 28-channel binary latent")
    T, H, W = 8, 64, 64
    mask_video = make_demo_mask_video(T=T, H=H, W=W, n_chars=2)
    print(f"    Input:  mask_video.shape = {mask_video.shape}  (T, H, W, 3)")

    latent = extract_and_compress_mask_to_latent(mask_video, spatial_factor=8)
    T_lat, H_lat, W_lat = latent.shape[1], latent.shape[2], latent.shape[3]
    print(f"    Output: latent.shape     = {latent.shape}  "
          f"({N_MASK_CHANNELS} colors × {TEMPORAL_STRIDE} temporal stride = {TOTAL_LATENT_CHANNELS} channels)")
    print(f"    Spatial compression:  {H}×{W} → {H_lat}×{W_lat}  ({8}× downsampling)")
    print(f"    Temporal compression: T={T} → T_lat={T_lat}  (stride {TEMPORAL_STRIDE})")

    # Verify channel activations
    ch_sums = latent.sum(axis=(1, 2, 3))  # (28,) total activation per channel
    color_sums = ch_sums.reshape(N_MASK_CHANNELS, TEMPORAL_STRIDE).sum(axis=1)
    print(f"\n    Channel activation per color (sum over all spatial+temporal positions):")
    for i, name in enumerate(COLOR_ORDER):
        bar = "█" * int(color_sums[i] * 2)
        print(f"    Ch{i} ({name:>6}): {color_sums[i]:6.1f}  {bar}")

    print(f"\n    Verification: K=6 binding slots + 1 env channel × 4 temporal = 28 ✓")
    print(f"    Model receives these 28 channels concatenated to VAE latent context.")

    visualize_mask_channels(mask_video, latent)

    # ── 2. Mode-Specific Shifted RoPE ─────────────────────────────────────
    print("\n[2] Mode-Specific Shifted RoPE")
    Tv, Hv, Wv = 16, 8, 14
    delta_W, delta_H = 16, 10
    print(f"    Video latent grid: Tv={Tv}, Hv={Hv}, Wv={Wv}")
    print(f"    Spatial offsets:   ΔW={delta_W} (driving stream), ΔH={delta_H} (ref in replacement)")

    anim = build_rope_coords(Tv, Hv, Wv, delta_W, delta_H, mode="animation")
    repl = build_rope_coords(Tv, Hv, Wv, delta_W, delta_H, mode="replacement")

    print_rope_table(anim, "animation")
    print_rope_table(repl, "replacement")

    analysis = token_overlap_analysis(anim, repl)
    print("\n    Disambiguation analysis:")
    for mode_name, r in analysis.items():
        status = "✓ DISJOINT" if r["is_disjoint"] else "✗ OVERLAP"
        print(f"    {mode_name:>12}: disambiguated via {r['disambiguation']:22}  "
              f"ref/video coord overlap = {r['t_overlap'] if mode_name == 'animation' else r['spatial_h_overlap']} tokens  {status}")

    # Show RoPE encoding distance between ref and video tokens
    print("\n    RoPE encoding: ref vs. video token angular distance (temporal dim)")
    for mode_name, coords in [("animation", anim), ("replacement", repl)]:
        ref_enc  = rope_1d_encoding(coords["ref"].t,   dim=16)
        vid_enc0 = rope_1d_encoding(coords["video"].t[:1], dim=16)
        dist = float(np.linalg.norm(ref_enc[0] - vid_enc0[0]))
        print(f"    {mode_name:>12}: ||enc(ref.t) - enc(video.t[0])|| = {dist:.4f}")

    print("\n    Key insight: animation mode separates streams TEMPORALLY (t=0 vs t≥1)")
    print("    Replacement mode separates streams SPATIALLY (ref gets extra ΔH offset)")
    print("    → prevents training conflicts between the two modes (ablation Fig.9c)")

    # ── 3. Bias-Aware DPO ─────────────────────────────────────────────────
    print("\n[3] Bias-Aware DPO Preference Dataset Construction")
    print("    Constructing (driving, ref, positive, negative) preference pairs...")
    print("    Negative = extra round of pose estimation error (esp. in finger joints)")

    pairs = construct_preference_dataset(n_samples=20, seed=0, use_weaker_estimator=True)

    hand_gaps = [p["hand_gap"] for p in pairs]
    body_gaps = [p["body_gap"] for p in pairs]
    neg_worse_frac = sum(p["neg_worse"] for p in pairs) / len(pairs)

    print(f"\n    Results over {len(pairs)} preference pairs:")
    print(f"    Fraction where r⁻ is worse than r:          {neg_worse_frac:.1%}  (should be ~100%)")
    print(f"    Mean hand MPJPE gap (r⁻ vs r):              {np.mean(hand_gaps):.4f}  (should be > 0)")
    print(f"    Mean body MPJPE gap (r⁻ vs r):              {np.mean(body_gaps):.4f}")
    print(f"    Ratio hand_gap / body_gap:                  {np.mean(hand_gaps)/max(np.mean(body_gaps),1e-6):.2f}x")
    print(f"    → Finger error accumulates disproportionately ({np.mean(hand_gaps)/max(np.mean(body_gaps),1e-6):.1f}× body)")

    # Show per-sample errors for first 5 pairs
    print("\n    Sample-level error comparison (first 5 pairs):")
    print(f"    {'ID':>3}  {'r pos hand':>12}  {'r⁻ neg hand':>12}  {'gap':>8}  {'neg_worse':>10}")
    print(f"    {'─'*54}")
    for p in pairs[:5]:
        better = "✓" if p["neg_worse"] else "✗"
        print(f"    {p['sample_id']:>3}  {p['pos_error']['hand_mpjpe']:>12.4f}  "
              f"{p['neg_error']['hand_mpjpe']:>12.4f}  {p['hand_gap']:>8.4f}  {better:>10}")

    print(f"\n    DPO preference tuple structure:")
    print(f"    (s=driving_video, R1=ref_frame, r=positive, r⁻=negative)")
    print(f"    Optimization: increase p(r|s,R1) / decrease p(r⁻|s,R1) via DPO loss")
    print(f"    Paper: trained for 400 steps on 64×H100, improves finger articulation")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Paper results (from Tab. 2 / Fig. 5–7, requires full 14B model):")
    print("  FVD on Studio-Bench:  287.11  (vs SCAIL 309.63, Wan-Animate 305.31)")
    print("  SSIM: 0.6453  PSNR: 19.09  LPIPS: 0.2231")
    print("  Motion Consistency win rate: 68.3% vs SCAIL, 65.0% vs Kling 3.0")
    print("  Multi-character Identity Isolation: 93.3% win vs MultiAnimate")
    print("  Replacement mode: beats MoCha (67.9% environment integration win rate)")
    print("  14B model, 64×H100, ~1 week training, MotionPair-60K dataset")
