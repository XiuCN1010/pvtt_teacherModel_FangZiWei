"""
Original WorldPlay Baseline Evaluation
========================================
Generate 3 scenes with unmodified WorldPlay-5B, evaluate DINO + PSNR.
All metric code is inline — no dependency on anti_drift_reward.py.

Produces: videos (.mp4), metrics (.json), drift curves (.png), frame grids (.png)

Usage:
    cd /data/fangziwei/anti-drift/HY-WorldPlay
    export PYTHONPATH=$(pwd):$(pwd)/wan:$PYTHONPATH
    export HF_ENDPOINT=https://hf-mirror.com
    CUDA_VISIBLE_DEVICES=1,2,3,5 torchrun --nproc_per_node=4 --master_port=29600 \
        /data/fangziwei/anti-drift/experiments/eval_original_worldplay.py
"""
import torch
import torch.nn.functional as F
import sys
import os
import time
import json
import argparse
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.join(os.path.abspath("."), "wan"))
from diffusers.utils import export_to_video

SCENES = {
    "beach":    {"prompt": "A beach with waves and sand",              "pose": "d-16,w-16", "seed": 42},
    "garden":   {"prompt": "A peaceful garden with flowers and trees", "pose": "w-16",      "seed": 42},
    "mountain": {"prompt": "A mountain landscape with snow",           "pose": "w-32",      "seed": 42},
}
OUTPUT_DIR = "/data/fangziwei/anti-drift/outputs/original_eval"


# ───── VIDEO GENERATION ─────
def generate_video(scene_name, scene_cfg, args):
    from wan.generate import WanRunner
    runner = WanRunner(
        model_id=args.model_id,
        ckpt_path=args.ckpt_path,
        ar_model_path=args.ar_model_path,
    )
    input_dict = {
        "prompt": scene_cfg["prompt"],
        "negative_prompt": (
            "色调艳丽,过曝,静态,细节模糊不清,字幕,风格,作品,画作,画面,静止,整体发灰,"
            "最差质量,低质量,JPEG压缩残留,丑陋的,残缺的,多余的手指,画得不好的手部,"
            "画得不好的脸部,畸形的,毁容的,形态畸形的肢体,手指融合,静止不动的画面,"
            "杂乱的背景,三条腿,背景人很多,倒着走"
        ),
        "num_frames": 961,
        "num_inference_steps": 50,
        "height": 704,
        "width": 1280,
        "image_path": None,
        "use_memory": True,
        "context_window_length": 16,
        "seed": scene_cfg["seed"],
        "pose": scene_cfg["pose"],
        "num_chunk": 4,
    }
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print(f"[GEN] {scene_name}: prompt={scene_cfg['prompt']!r}, pose={scene_cfg['pose']}")
    t0 = time.time()
    result = runner.predict(input_dict)
    elapsed = time.time() - t0
    if rank == 0 and result["video"] is not None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        vp = os.path.join(OUTPUT_DIR, f"{scene_name}.mp4")
        export_to_video(result["video"][0], vp, fps=16)
        print(f"[GEN] {scene_name}: saved {vp} ({elapsed:.1f}s, shape={result['video'].shape})")
        return vp
    return None


# ───── METRIC COMPUTATION ─────
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
        psnrs.append(100.0 if mse < 1e-10 else float(10 * np.log10(255 ** 2 / mse)))
    return {
        "psnr_per_frame": psnrs,
        "psnr_mean": float(np.mean(psnrs)),
        "psnr_min": float(np.min(psnrs)),
    }


# ───── VISUALIZATION ─────
def plot_drift_curves(all_m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = {"beach": "#1f77b4", "garden": "#2ca02c", "mountain": "#d62728"}

    # DINO Anchor Similarity
    ax = axes[0, 0]
    for n, m in all_m.items():
        v = m["dino"]["anchor_sim_per_frame"]
        ax.plot(range(len(v)), v, label=f"{n} (last={v[-1]:.4f})", color=colors[n], lw=2)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("DINO Anchor Similarity (vs Frame 0)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.75, 1.02)

    # DINO Adjacent Similarity
    ax = axes[0, 1]
    for n, m in all_m.items():
        v = m["dino"]["adj_sim_per_frame"]
        ax.plot(range(len(v)), v, label=f"{n} (mean={np.mean(v):.4f})", color=colors[n], lw=2)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("DINO Adjacent Frame Similarity", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Temporal PSNR
    ax = axes[1, 0]
    for n, m in all_m.items():
        v = m["psnr"]["psnr_per_frame"]
        ax.plot(range(len(v)), v, label=f"{n} (mean={np.mean(v):.1f}dB)", color=colors[n], lw=2)
    ax.set_xlabel("Frame")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Temporal PSNR (Adjacent Frames)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Summary bar chart
    ax = axes[1, 1]
    scenes = list(all_m.keys())
    x = np.arange(len(scenes))
    w = 0.25
    d_adj = [all_m[s]["dino"]["adj_sim_mean"] for s in scenes]
    d_anc = [all_m[s]["dino"]["anchor_sim_mean"] for s in scenes]
    p_n = [min(all_m[s]["psnr"]["psnr_mean"] / 30, 1.0) for s in scenes]
    b1 = ax.bar(x - w, d_adj, w, label="DINO Adj Sim", color="#4c72b0")
    b2 = ax.bar(x, d_anc, w, label="DINO Anchor Sim", color="#55a868")
    b3 = ax.bar(x + w, p_n, w, label="PSNR/30 (norm)", color="#c44e52")
    for bs in [b1, b2, b3]:
        for b in bs:
            ax.annotate(
                f"{b.get_height():.3f}",
                xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in scenes])
    ax.set_title("Summary (Original WorldPlay-5B)", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    pp = os.path.join(OUTPUT_DIR, "drift_curves.png")
    plt.savefig(pp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIS] Drift curves saved: {pp}")


def save_frame_grid(frames, scene_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(frames)
    indices = [int(i * (n - 1) / 7) for i in range(8)]
    fig, axes = plt.subplots(2, 4, figsize=(20, 6))
    for ax, fi in zip(axes.flat, indices):
        ax.imshow(frames[fi])
        ax.set_title(f"Frame {fi}")
        ax.axis("off")
    fig.suptitle(
        f"{scene_name.capitalize()} - Frame Samples (Original WorldPlay-5B)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    gp = os.path.join(OUTPUT_DIR, f"{scene_name}_frame_grid.png")
    plt.savefig(gp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIS] Frame grid saved: {gp}")


# ───── MAIN ─────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="/data/fangziwei/models/Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--ar_model_path", default="/data/fangziwei/models/HY-WorldPlay/wan_transformer")
    parser.add_argument("--ckpt_path", default="/data/fangziwei/models/HY-WorldPlay/wan_distilled_model/model.pt")
    parser.add_argument("--skip_gen", action="store_true", help="Skip generation, only evaluate existing videos")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Generate videos
    video_paths = {}
    for sn, sc in SCENES.items():
        vp = os.path.join(OUTPUT_DIR, f"{sn}.mp4")
        if args.skip_gen and os.path.exists(vp):
            if rank == 0:
                print(f"[SKIP] {sn}.mp4 exists, skipping generation")
            video_paths[sn] = vp
        else:
            r = generate_video(sn, sc, args)
            if r:
                video_paths[sn] = r

    if rank != 0:
        return

    # Step 2: Evaluate metrics
    print(f"\n{'=' * 70}")
    print("EVALUATING METRICS (standalone DINO + PSNR)")
    print(f"{'=' * 70}\n")

    all_m = {}
    for sn in SCENES:
        vp = video_paths.get(sn)
        if not vp or not os.path.exists(vp):
            print(f"[WARN] No video for {sn}")
            continue
        print(f"\n-- {sn.upper()} --")
        frames = extract_frames(vp)

        print("  Computing DINO metrics...")
        dino = compute_dino_metrics(frames, device="cuda:0")

        print("  Computing PSNR metrics...")
        psnr = compute_psnr_metrics(frames)

        composite = (
            0.5 * (0.6 * dino["adj_sim_mean"] + 0.4 * dino["anchor_sim_mean"])
            + 0.3 * min((psnr["psnr_mean"] - 15) / 25, 1.0)
            + 0.2 * 0.5
        )

        m = {
            "scene": sn,
            "model": "WorldPlay-5B (original, no optimization)",
            "num_frames": len(frames),
            "num_chunks": 4,
            "resolution": "1280x704",
            "fps": 16,
            "dino": dino,
            "psnr": psnr,
            "composite": float(composite),
        }
        all_m[sn] = m

        mpath = os.path.join(OUTPUT_DIR, f"{sn}_metrics.json")
        with open(mpath, "w") as f:
            json.dump(m, f, indent=2)

        save_frame_grid(frames, sn)

    # Step 3: Print summary
    print(f"\n{'=' * 82}")
    print(f"RESULTS: Original WorldPlay-5B Baseline (4 chunks, 61 frames, 1280x704, 16fps)")
    print(f"{'=' * 82}")
    header = f"{'Scene':<12} {'DINO Adj':>10} {'DINO Anchor':>13} {'Anchor Drop':>13} {'PSNR Mean':>11} {'PSNR Min':>10} {'Composite':>11}"
    print(header)
    print("-" * 82)
    for n in ["beach", "garden", "mountain"]:
        m = all_m.get(n)
        if not m:
            continue
        d = m["dino"]
        p = m["psnr"]
        drop = f"-{d['anchor_sim_drop_pct']:.1f}%"
        print(
            f"{n.capitalize():<12} {d['adj_sim_mean']:>10.4f} "
            f"{d['anchor_sim_mean']:>13.4f} {drop:>13} "
            f"{p['psnr_mean']:>10.2f}dB {p['psnr_min']:>9.2f}dB "
            f"{m['composite']:>11.4f}"
        )
    print("-" * 82)

    # Step 4: Plot drift curves
    if len(all_m) >= 2:
        plot_drift_curves(all_m)

    # Step 5: Save summary
    summary = {
        "experiment": "Original WorldPlay-5B Baseline Evaluation",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "WorldPlay-5B (WAN pipeline, distilled 4-step)",
        "hardware": "RTX 5090 x4 (32GB each)",
        "resolution": "1280x704",
        "fps": 16,
        "num_chunks": 4,
        "num_frames": 61,
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
    spath = os.path.join(OUTPUT_DIR, "summary.json")
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] All results saved to {OUTPUT_DIR}/")
    print(f"  Videos:      *.mp4")
    print(f"  Metrics:     *_metrics.json")
    print(f"  Frame grids: *_frame_grid.png")
    print(f"  Drift plot:  drift_curves.png")
    print(f"  Summary:     summary.json")


if __name__ == "__main__":
    main()
