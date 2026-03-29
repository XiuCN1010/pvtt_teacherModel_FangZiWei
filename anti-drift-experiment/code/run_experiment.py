"""
Anti-Drift Experiment: Baseline vs Post-Hoc Optimization
=========================================================
Generates 3 scenes (Garden, Beach, Mountain) with WorldPlay-5B (WAN pipeline),
then compares baseline generation vs Best-of-N post-hoc optimization using
DINO consistency + Temporal PSNR rewards.

Usage:
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 experiments/run_experiment.py \
        --mode baseline --scene garden

    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 experiments/run_experiment.py \
        --mode optimized --scene garden --num_candidates 3
"""

import torch
import sys
import os
import time
import argparse
import json
import numpy as np
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "HY-WorldPlay"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "HY-WorldPlay", "wan"))

from diffusers.utils import export_to_video

SCENES = {
    "garden": {
        "prompt": "A peaceful garden with flowers and trees, sunlight filtering through green leaves, stone pathway winding between colorful flower beds",
        "pose": "w-16",
        "seed": 42,
    },
    "beach": {
        "prompt": "A beach with waves and sand, ocean waves gently rolling onto golden sandy shore, clear blue sky with scattered clouds",
        "pose": "d-16,w-16",
        "seed": 42,
    },
    "mountain": {
        "prompt": "A mountain landscape with snow, majestic snow-capped peaks against deep blue sky, pine trees in foreground",
        "pose": "w-32",
        "seed": 42,
    },
}


def build_runner(args):
    from wan.generate import WanRunner
    return WanRunner(
        model_id=args.model_id,
        ckpt_path=args.ckpt_path,
        ar_model_path=args.ar_model_path,
    )


def generate_baseline(runner, scene_config, args):
    input_dict = {
        "prompt": scene_config["prompt"],
        "negative_prompt": "色调艳丽,过曝,静态,细节模糊不清,字幕,最差质量,低质量",
        "num_frames": 961,
        "num_inference_steps": 50,
        "height": 704,
        "width": 1280,
        "image_path": None,
        "use_memory": True,
        "context_window_length": 16,
        "seed": scene_config["seed"],
        "pose": scene_config["pose"],
        "num_chunk": args.num_chunks,
    }
    return runner.predict(input_dict)


def evaluate_video(video_array, scene_config, device="cuda:0"):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from anti_drift_reward import CompositeReward, frames_from_video_array

    frames = frames_from_video_array(video_array)
    reward_model = CompositeReward(device=device, use_hps=False)
    reward_model.set_anchor(frames[0])

    full_result = reward_model.score(frames, scene_config["prompt"])

    chunk_size = 16
    chunk_results = []
    for i in range(0, len(frames), chunk_size):
        chunk_frames = frames[i : i + chunk_size]
        if len(chunk_frames) < 2:
            continue
        chunk_results.append(
            reward_model.score(chunk_frames, scene_config["prompt"])
        )

    return {
        "full": full_result,
        "per_chunk": chunk_results,
        "num_frames": len(frames),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "optimized"], default="baseline")
    parser.add_argument("--scene", choices=["garden", "beach", "mountain", "all"], default="all")
    parser.add_argument("--num_chunks", type=int, default=4)
    parser.add_argument("--num_candidates", type=int, default=3)
    parser.add_argument("--model_id", type=str, default="/data/fangziwei/models/Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--ar_model_path", type=str, default="/data/fangziwei/models/HY-WorldPlay/wan_transformer")
    parser.add_argument("--ckpt_path", type=str, default="/data/fangziwei/models/HY-WorldPlay/wan_distilled_model/model.pt")
    parser.add_argument("--output_dir", type=str, default="/data/fangziwei/anti-drift/outputs")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    os.makedirs(args.output_dir, exist_ok=True)

    scenes_to_run = list(SCENES.keys()) if args.scene == "all" else [args.scene]
    runner = build_runner(args)

    for scene_name in scenes_to_run:
        scene_config = SCENES[scene_name]
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Scene: {scene_name} | Mode: {args.mode}")
            print(f"Prompt: {scene_config['prompt']}")
            print(f"{'='*60}\n")

        start_time = time.time()
        result = generate_baseline(runner, scene_config, args)
        gen_time = time.time() - start_time

        if rank == 0 and result["video"] is not None:
            video_path = os.path.join(args.output_dir, f"{args.mode}_{scene_name}.mp4")
            export_to_video(result["video"][0], video_path, fps=16)
            print(f"Video saved: {video_path} ({gen_time:.1f}s)")

            print(f"Evaluating {scene_name}...")
            eval_result = evaluate_video(result["video"], scene_config, device=f"cuda:{runner.local_rank}")

            metrics = {
                "scene": scene_name,
                "mode": args.mode,
                "num_chunks": args.num_chunks,
                "generation_time_s": gen_time,
                "num_frames": eval_result["num_frames"],
                "full_dino_adj_sim": eval_result["full"]["dino"]["adj_sim"],
                "full_dino_anchor_sim": eval_result["full"]["dino"]["anchor_sim"],
                "full_dino_reward": eval_result["full"]["dino"]["reward"],
                "full_psnr_mean": eval_result["full"]["psnr"]["mean_psnr"],
                "full_psnr_min": eval_result["full"]["psnr"]["min_psnr"],
                "full_composite": eval_result["full"]["composite_reward"],
                "per_frame_anchor_sim": eval_result["full"]["dino"]["per_frame_anchor_sim"],
                "per_frame_psnr": eval_result["full"]["psnr"]["per_frame_psnr"],
            }

            metrics_path = os.path.join(args.output_dir, f"{args.mode}_{scene_name}_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            print(f"\n--- {scene_name} ({args.mode}) ---")
            print(f"  DINO Adjacent Sim:  {metrics['full_dino_adj_sim']:.4f}")
            print(f"  DINO Anchor Sim:    {metrics['full_dino_anchor_sim']:.4f}")
            print(f"  PSNR Mean:          {metrics['full_psnr_mean']:.2f} dB")
            print(f"  Composite:          {metrics['full_composite']:.4f}")


if __name__ == "__main__":
    main()
