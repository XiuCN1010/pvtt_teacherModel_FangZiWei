"""
DINO-Reward Optimized WorldPlay Evaluation
============================================
Generate 3 scenes with WorldPlay-5B using Best-of-N chunk selection
guided by DINO consistency + Temporal PSNR rewards.

For each chunk (after the first), generate N candidates with different
random seeds, score with DINO+PSNR composite reward, select the best.

All metric code inline — no dependency on anti_drift_reward.py.

Usage:
    cd /data/fangziwei/anti-drift/HY-WorldPlay
    export PYTHONPATH=$(pwd):$(pwd)/wan:$PYTHONPATH
    export HF_ENDPOINT=https://hf-mirror.com
    CUDA_VISIBLE_DEVICES=1,2,3,5 torchrun --nproc_per_node=4 --master_port=29700 \
        /data/fangziwei/anti-drift/experiments/eval_dino_optimized_worldplay.py
"""
import torch
import torch.nn.functional as F
import sys
import os
import time
import json
import argparse
import numpy as np
from copy import deepcopy
from PIL import Image

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.join(os.path.abspath("."), "wan"))
from diffusers.utils import export_to_video

SCENES = {
    "beach":    {"prompt": "A beach with waves and sand",              "pose": "d-16,w-16", "seed": 42},
    "garden":   {"prompt": "A peaceful garden with flowers and trees", "pose": "w-16",      "seed": 42},
    "mountain": {"prompt": "A mountain landscape with snow",           "pose": "w-32",      "seed": 42},
}
OUTPUT_DIR = "/data/fangziwei/anti-drift/outputs/dino_optimized_eval"
NUM_CANDIDATES = 3  # Best-of-N


# ═══════════════════════════════════════════════════
# INLINE DINO + PSNR REWARD (no external dependency)
# ═══════════════════════════════════════════════════
class InlineDINOScorer:
    """Lightweight DINO scorer for Best-of-N selection during generation."""
    def __init__(self, device="cuda"):
        from transformers import AutoImageProcessor, AutoModel
        self.proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
        self.device = device
        self.anchor_feat = None

    @torch.no_grad()
    def _get_feats(self, frames):
        inp = self.proc(images=frames, return_tensors="pt").to(self.device)
        out = self.model(**inp)
        return F.normalize(out.last_hidden_state[:, 0, :], dim=-1)

    def set_anchor(self, frame):
        self.anchor_feat = self._get_feats([frame]).cpu()

    def score_chunk(self, chunk_frames, prev_last_frame=None):
        """Score a chunk of frames. Returns composite reward scalar."""
        eval_frames = []
        if prev_last_frame is not None:
            eval_frames.append(prev_last_frame)
        eval_frames.extend(chunk_frames)

        feats = self._get_feats(eval_frames).cpu()

        # Adjacent similarity
        adj_sim = (feats[:-1] * feats[1:]).sum(dim=-1).mean().item()

        # Anchor similarity (vs first frame of entire video)
        if self.anchor_feat is not None:
            anchor_sim = (feats * self.anchor_feat).sum(dim=-1).mean().item()
        else:
            anchor_sim = adj_sim

        # PSNR between consecutive frames
        arrs = [np.array(f, dtype=np.float64) for f in eval_frames]
        psnrs = []
        for i in range(1, len(arrs)):
            mse = np.mean((arrs[i] - arrs[i - 1]) ** 2)
            psnrs.append(100.0 if mse < 1e-10 else float(10 * np.log10(255**2 / mse)))
        mean_psnr = float(np.mean(psnrs))
        psnr_norm = min((mean_psnr - 15) / 25, 1.0)

        # Composite: DINO (0.6 adj + 0.4 anchor) * 0.5 + PSNR * 0.3 + 0.1
        dino_score = 0.6 * adj_sim + 0.4 * anchor_sim
        composite = 0.5 * dino_score + 0.3 * psnr_norm + 0.2 * 0.5

        return {
            "composite": composite,
            "dino_adj": adj_sim,
            "dino_anchor": anchor_sim,
            "psnr_mean": mean_psnr,
        }


# ═══════════════════════════════════════════════════
# VIDEO GENERATION WITH BEST-OF-N DINO SELECTION
# ═══════════════════════════════════════════════════
def video_array_to_pil(video_np):
    """Convert video numpy [1, T, H, W, C] or [T, H, W, C] to list of PIL."""
    if video_np.ndim == 5:
        video_np = video_np[0]
    out = []
    for i in range(video_np.shape[0]):
        f = video_np[i]
        if f.max() <= 1.0:
            f = (f * 255).astype(np.uint8)
        else:
            f = f.astype(np.uint8)
        out.append(Image.fromarray(f))
    return out


def generate_dino_optimized(scene_name, scene_cfg, args):
    """Generate video with Best-of-N DINO-guided chunk selection."""
    from wan.generate import WanRunner
    from wan.inference.helper import CHUNK_SIZE
    from hyvideo.generate import pose_string_to_json, pose_to_input

    rank = int(os.environ.get("RANK", 0))

    runner = WanRunner(
        model_id=args.model_id,
        ckpt_path=args.ckpt_path,
        ar_model_path=args.ar_model_path,
    )
    pipe = runner.pipe

    # Initialize DINO scorer on rank 0
    scorer = None
    if rank == 0:
        scorer = InlineDINOScorer(device=f"cuda:{runner.local_rank}")

    prompt = scene_cfg["prompt"]
    neg_prompt = (
        "色调艳丽,过曝,静态,细节模糊不清,字幕,风格,作品,画作,画面,静止,整体发灰,"
        "最差质量,低质量,JPEG压缩残留,丑陋的,残缺的,多余的手指,画得不好的手部,"
        "画得不好的脸部,畸形的,毁容的,形态畸形的肢体,手指融合,静止不动的画面,"
        "杂乱的背景,三条腿,背景人很多,倒着走"
    )
    seed = scene_cfg["seed"]
    pose = scene_cfg["pose"]
    num_chunk = 4

    # Parse pose
    pose_json = pose_string_to_json(pose)
    all_viewmats, all_Ks, all_action = pose_to_input(pose_json, len(pose_json))

    run_args = dict(
        prompt=prompt, negative_prompt=neg_prompt,
        height=704, width=1280, num_frames=961, num_inference_steps=50,
        guidance_scale=1.0, few_step=True, first_chunk_size=CHUNK_SIZE,
        return_dict=False, image_path=None,
        use_memory=True, context_window_length=16,
    )

    all_video = []
    selection_log = []

    for chunk_i in range(num_chunk):
        start_idx = chunk_i * CHUNK_SIZE
        end_idx = start_idx + CHUNK_SIZE
        curr_viewmats = all_viewmats[start_idx:end_idx]
        curr_Ks = all_Ks[start_idx:end_idx]
        curr_action = all_action[start_idx:end_idx]

        if chunk_i == 0:
            # First chunk: generate normally (no selection needed)
            torch.manual_seed(seed)
            pipe(
                **run_args, chunk_i=chunk_i,
                viewmats=curr_viewmats.unsqueeze(0),
                Ks=curr_Ks.unsqueeze(0),
                action=curr_action.unsqueeze(0),
                output_type="latent",
            )
            for lat_i in range(4):
                v = pipe.decode_next_latent(output_type="np")
                all_video.append(v)

            # Set DINO anchor to first frame
            if rank == 0:
                first_frames = video_array_to_pil(all_video[0])
                scorer.set_anchor(first_frames[0])
                print(f"  [Chunk 0] Generated (anchor set)")

        else:
            # Best-of-N selection for subsequent chunks
            # Save pipeline state using tensor .clone() (deepcopy fails in distributed)
            saved_latents = pipe.ctx["latents"].clone()
            saved_kv = []
            for layer_kv in pipe._kv_cache:
                saved_kv.append({
                    "k": layer_kv["k"].clone() if layer_kv["k"] is not None else None,
                    "v": layer_kv["v"].clone() if layer_kv["v"] is not None else None,
                })

            best_score = -float("inf")
            best_latents = None
            best_kv_state = None
            best_frames = None
            best_info = None
            best_cand_i = 0

            # Also save VAE decoder cache (temporal consistency state)
            saved_vae_cache = [t.clone() if t is not None else None for t in pipe.vae._feat_map]

            cand_results = []

            for cand_i in range(NUM_CANDIDATES):
                # Restore pipeline latents
                pipe.ctx["latents"].copy_(saved_latents)
                for i, layer_kv in enumerate(pipe._kv_cache):
                    if saved_kv[i]["k"] is not None:
                        layer_kv["k"] = saved_kv[i]["k"].clone()
                        layer_kv["v"] = saved_kv[i]["v"].clone()
                    else:
                        layer_kv["k"] = None
                        layer_kv["v"] = None
                pipe.ctx["kv_cache"] = pipe._kv_cache

                # Restore VAE cache to post-previous-chunk state
                for vi, t in enumerate(saved_vae_cache):
                    pipe.vae._feat_map[vi] = t.clone() if t is not None else None

                cand_seed = seed + chunk_i * 1000 + cand_i
                torch.manual_seed(cand_seed)

                new_noise = torch.randn(
                    pipe.ctx["latents"][:, :, start_idx:end_idx].shape,
                    device=pipe.ctx["latents"].device,
                    dtype=pipe.ctx["latents"].dtype,
                )
                pipe.ctx["latents"][:, :, start_idx:end_idx] = new_noise

                # Generate (ALL ranks)
                pipe(
                    **run_args, chunk_i=chunk_i,
                    viewmats=curr_viewmats.unsqueeze(0),
                    Ks=curr_Ks.unsqueeze(0),
                    action=curr_action.unsqueeze(0),
                    output_type="latent",
                )

                # Decode (ALL ranks) — only decode last latent to score, save full latents
                post_latents = pipe.ctx["latents"][:, :, start_idx:end_idx].clone()

                # Decode only 1 representative latent (last) for scoring to save memory
                # We skip full decode here; decode fully only for the winner
                # Instead, decode all 4 for proper scoring
                cand_frames = []
                for lat_i in range(4):
                    v = pipe.decode_next_latent(output_type="np")
                    cand_frames.append(v)

                score_info = None
                score = -float("inf")
                if rank == 0:
                    cand_video = np.concatenate(cand_frames, axis=1)
                    cand_pil = video_array_to_pil(cand_video)
                    prev_last = video_array_to_pil(all_video[-1])[-1]
                    score_info = scorer.score_chunk(cand_pil, prev_last_frame=prev_last)
                    score = score_info["composite"]
                    print(f"  [Chunk {chunk_i}] Candidate {cand_i}: "
                          f"composite={score:.4f}, dino_adj={score_info['dino_adj']:.4f}, "
                          f"dino_anchor={score_info['dino_anchor']:.4f}, psnr={score_info['psnr_mean']:.1f}dB")
                    del cand_video, cand_pil  # free memory

                cand_results.append((cand_seed, post_latents, cand_frames, score, score_info))
                torch.cuda.empty_cache()

            # Select best on rank 0, broadcast to all
            if rank == 0:
                best_idx = max(range(len(cand_results)), key=lambda i: cand_results[i][3])
            else:
                best_idx = 0
            best_idx_tensor = torch.tensor([best_idx], device=f"cuda:{runner.local_rank}")
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(best_idx_tensor, src=0)
            chosen = int(best_idx_tensor.item())

            # Restore full state for the winner re-run
            pipe.ctx["latents"].copy_(saved_latents)
            for i, layer_kv in enumerate(pipe._kv_cache):
                if saved_kv[i]["k"] is not None:
                    layer_kv["k"] = saved_kv[i]["k"].clone()
                    layer_kv["v"] = saved_kv[i]["v"].clone()
                else:
                    layer_kv["k"] = None
                    layer_kv["v"] = None
            pipe.ctx["kv_cache"] = pipe._kv_cache
            # Restore VAE cache
            for vi, t in enumerate(saved_vae_cache):
                pipe.vae._feat_map[vi] = t.clone() if t is not None else None

            torch.manual_seed(cand_results[chosen][0])
            new_noise = torch.randn(
                pipe.ctx["latents"][:, :, start_idx:end_idx].shape,
                device=pipe.ctx["latents"].device,
                dtype=pipe.ctx["latents"].dtype,
            )
            pipe.ctx["latents"][:, :, start_idx:end_idx] = new_noise

            pipe(
                **run_args, chunk_i=chunk_i,
                viewmats=curr_viewmats.unsqueeze(0),
                Ks=curr_Ks.unsqueeze(0),
                action=curr_action.unsqueeze(0),
                output_type="latent",
            )
            final_frames = []
            for lat_i in range(4):
                v = pipe.decode_next_latent(output_type="np")
                final_frames.append(v)
            all_video.extend(final_frames)

            if rank == 0:
                best_info = cand_results[chosen][4]
                best_score = cand_results[chosen][3]
                print(f"  [Chunk {chunk_i}] SELECTED candidate {chosen}: composite={best_score:.4f}")
                selection_log.append(best_info)

            # Cleanup
            del cand_results, saved_vae_cache

            # Free saved states
            del saved_latents, saved_kv
            torch.cuda.empty_cache()

    if rank == 0 and all_video:
        video = np.concatenate(all_video, axis=1)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        vp = os.path.join(OUTPUT_DIR, f"{scene_name}.mp4")
        export_to_video(video[0], vp, fps=16)
        print(f"  [SAVED] {vp} (shape={video.shape})")
        return vp, selection_log
    return None, []


# ═══════════════════════════════════════════════════
# POST-HOC EVALUATION (same as original eval)
# ═══════════════════════════════════════════════════
def extract_frames(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    print(f"  Extracted {len(frames)} frames from {os.path.basename(video_path)}")
    return frames


def compute_dino_metrics(frames, device="cuda"):
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(frames), 16):
            batch = frames[i:i + 16]
            inp = proc(images=batch, return_tensors="pt").to(device)
            out = model(**inp)
            f = F.normalize(out.last_hidden_state[:, 0, :], dim=-1)
            all_feats.append(f.cpu())
    feats = torch.cat(all_feats, dim=0)
    adj = (feats[:-1] * feats[1:]).sum(dim=-1)
    anchor = (feats * feats[0:1]).sum(dim=-1)
    del model
    torch.cuda.empty_cache()
    return {
        "adj_sim_per_frame": adj.numpy().tolist(),
        "adj_sim_mean": float(adj.mean()),
        "anchor_sim_per_frame": anchor.numpy().tolist(),
        "anchor_sim_mean": float(anchor.mean()),
        "anchor_sim_last": float(anchor[-1]),
        "anchor_sim_drop_pct": float((1.0 - anchor[-1]) * 100),
    }


def compute_psnr_metrics(frames):
    arrs = [np.array(f, dtype=np.float64) for f in frames]
    psnrs = []
    for i in range(1, len(arrs)):
        mse = np.mean((arrs[i] - arrs[i - 1]) ** 2)
        psnrs.append(100.0 if mse < 1e-10 else float(10 * np.log10(255**2 / mse)))
    return {
        "psnr_per_frame": psnrs,
        "psnr_mean": float(np.mean(psnrs)),
        "psnr_min": float(np.min(psnrs)),
    }


# ═══════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════
def plot_drift_curves(all_m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = {"beach": "#1f77b4", "garden": "#2ca02c", "mountain": "#d62728"}

    ax = axes[0, 0]
    for n, m in all_m.items():
        v = m["dino"]["anchor_sim_per_frame"]
        ax.plot(range(len(v)), v, label=f"{n} (last={v[-1]:.4f})", color=colors[n], lw=2)
    ax.set_xlabel("Frame"); ax.set_ylabel("Cosine Similarity")
    ax.set_title("DINO Anchor Similarity (vs Frame 0) — DINO Optimized", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0.75, 1.02)

    ax = axes[0, 1]
    for n, m in all_m.items():
        v = m["dino"]["adj_sim_per_frame"]
        ax.plot(range(len(v)), v, label=f"{n} (mean={np.mean(v):.4f})", color=colors[n], lw=2)
    ax.set_xlabel("Frame"); ax.set_ylabel("Cosine Similarity")
    ax.set_title("DINO Adjacent Frame Similarity — DINO Optimized", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for n, m in all_m.items():
        v = m["psnr"]["psnr_per_frame"]
        ax.plot(range(len(v)), v, label=f"{n} (mean={np.mean(v):.1f}dB)", color=colors[n], lw=2)
    ax.set_xlabel("Frame"); ax.set_ylabel("PSNR (dB)")
    ax.set_title("Temporal PSNR — DINO Optimized", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    scenes = list(all_m.keys()); x = np.arange(len(scenes)); w = 0.25
    d_adj = [all_m[s]["dino"]["adj_sim_mean"] for s in scenes]
    d_anc = [all_m[s]["dino"]["anchor_sim_mean"] for s in scenes]
    p_n = [min(all_m[s]["psnr"]["psnr_mean"] / 30, 1.0) for s in scenes]
    b1 = ax.bar(x - w, d_adj, w, label="DINO Adj Sim", color="#4c72b0")
    b2 = ax.bar(x, d_anc, w, label="DINO Anchor Sim", color="#55a868")
    b3 = ax.bar(x + w, p_n, w, label="PSNR/30 (norm)", color="#c44e52")
    for bs in [b1, b2, b3]:
        for b in bs:
            ax.annotate(f"{b.get_height():.3f}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([s.capitalize() for s in scenes])
    ax.set_title("Summary (DINO-Optimized WorldPlay-5B)", fontweight="bold")
    ax.legend(); ax.set_ylim(0, 1.1); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    pp = os.path.join(OUTPUT_DIR, "drift_curves.png")
    plt.savefig(pp, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[VIS] {pp}")


def save_frame_grid(frames, scene_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(frames); indices = [int(i * (n - 1) / 7) for i in range(8)]
    fig, axes = plt.subplots(2, 4, figsize=(20, 6))
    for ax, fi in zip(axes.flat, indices):
        ax.imshow(frames[fi]); ax.set_title(f"Frame {fi}"); ax.axis("off")
    fig.suptitle(f"{scene_name.capitalize()} - Frame Samples (DINO-Optimized)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    gp = os.path.join(OUTPUT_DIR, f"{scene_name}_frame_grid.png")
    plt.savefig(gp, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[VIS] {gp}")


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="/data/fangziwei/models/Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--ar_model_path", default="/data/fangziwei/models/HY-WorldPlay/wan_transformer")
    parser.add_argument("--ckpt_path", default="/data/fangziwei/models/HY-WorldPlay/wan_distilled_model/model.pt")
    parser.add_argument("--skip_gen", action="store_true")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Generate videos with DINO Best-of-N
    video_paths = {}
    all_selection_logs = {}
    for sn, sc in SCENES.items():
        vp = os.path.join(OUTPUT_DIR, f"{sn}.mp4")
        if args.skip_gen and os.path.exists(vp):
            if rank == 0:
                print(f"[SKIP] {sn}.mp4 exists")
            video_paths[sn] = vp
        else:
            if rank == 0:
                print(f"\n{'=' * 60}")
                print(f"GENERATING: {sn} (Best-of-{NUM_CANDIDATES} DINO selection)")
                print(f"{'=' * 60}")
            r, slog = generate_dino_optimized(sn, sc, args)
            if r:
                video_paths[sn] = r
                all_selection_logs[sn] = slog

    if rank != 0:
        return

    # Step 2: Evaluate
    print(f"\n{'=' * 70}")
    print(f"EVALUATING DINO-OPTIMIZED RESULTS")
    print(f"{'=' * 70}\n")

    all_m = {}
    for sn in SCENES:
        vp = video_paths.get(sn)
        if not vp or not os.path.exists(vp):
            continue
        print(f"\n-- {sn.upper()} --")
        frames = extract_frames(vp)
        print("  DINO metrics...")
        dino = compute_dino_metrics(frames, device="cuda:0")
        print("  PSNR metrics...")
        psnr = compute_psnr_metrics(frames)
        composite = (
            0.5 * (0.6 * dino["adj_sim_mean"] + 0.4 * dino["anchor_sim_mean"])
            + 0.3 * min((psnr["psnr_mean"] - 15) / 25, 1.0)
            + 0.2 * 0.5
        )
        m = {
            "scene": sn,
            "model": "WorldPlay-5B + DINO Best-of-3 Optimization",
            "num_frames": len(frames),
            "num_chunks": 4,
            "num_candidates": NUM_CANDIDATES,
            "resolution": "1280x704",
            "fps": 16,
            "dino": dino,
            "psnr": psnr,
            "composite": float(composite),
            "selection_log": all_selection_logs.get(sn, []),
        }
        all_m[sn] = m
        with open(os.path.join(OUTPUT_DIR, f"{sn}_metrics.json"), "w") as f:
            json.dump(m, f, indent=2)
        save_frame_grid(frames, sn)

    # Step 3: Print summary
    print(f"\n{'=' * 82}")
    print(f"RESULTS: DINO-Optimized WorldPlay-5B (Best-of-{NUM_CANDIDATES}, 4 chunks, 61 frames)")
    print(f"{'=' * 82}")
    print(f"{'Scene':<12} {'DINO Adj':>10} {'DINO Anchor':>13} {'Anchor Drop':>13} {'PSNR Mean':>11} {'PSNR Min':>10} {'Composite':>11}")
    print("-" * 82)
    for n in ["beach", "garden", "mountain"]:
        m = all_m.get(n)
        if not m:
            continue
        d, p = m["dino"], m["psnr"]
        drop = f"-{d['anchor_sim_drop_pct']:.1f}%"
        print(f"{n.capitalize():<12} {d['adj_sim_mean']:>10.4f} "
              f"{d['anchor_sim_mean']:>13.4f} {drop:>13} "
              f"{p['psnr_mean']:>10.2f}dB {p['psnr_min']:>9.2f}dB "
              f"{m['composite']:>11.4f}")
    print("-" * 82)

    if len(all_m) >= 2:
        plot_drift_curves(all_m)

    # Save summary
    summary = {
        "experiment": "DINO-Optimized WorldPlay-5B (Best-of-3)",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": f"Best-of-{NUM_CANDIDATES} chunk selection using DINO+PSNR reward",
        "model": "WorldPlay-5B (WAN pipeline, distilled 4-step)",
        "hardware": "RTX 5090 x4",
        "scenes": {},
    }
    for n in ["beach", "garden", "mountain"]:
        m = all_m.get(n, {})
        if m:
            summary["scenes"][n] = {
                "dino_adj_sim": m["dino"]["adj_sim_mean"],
                "dino_anchor_sim": m["dino"]["anchor_sim_mean"],
                "dino_anchor_drop": f"-{m['dino']['anchor_sim_drop_pct']:.1f}%",
                "psnr_mean": m["psnr"]["psnr_mean"],
                "psnr_min": m["psnr"]["psnr_min"],
                "composite": m["composite"],
            }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] All results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
