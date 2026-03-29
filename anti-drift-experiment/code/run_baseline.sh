#!/bin/bash
# Run baseline experiments on free GPUs (0, 2, 3, 5)
# Each scene uses 1 GPU for WAN pipeline inference

cd /data/fangziwei/anti-drift/HY-WorldPlay
export PYTHONPATH=$(pwd):$(pwd)/wan:$PYTHONPATH

# Download DINOv2 model for reward computation
export HF_ENDPOINT=https://hf-mirror.com

echo "============================================"
echo "Starting Anti-Drift Baseline Experiments"
echo "Using GPUs: 0, 2, 3 (leaving 5 as spare)"
echo "============================================"

# Run 3 scenes in parallel on different GPUs
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 \
    /data/fangziwei/anti-drift/experiments/run_experiment.py \
    --mode baseline --scene garden --num_chunks 4 \
    > /data/fangziwei/anti-drift/outputs/log_baseline_garden.txt 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=29501 \
    /data/fangziwei/anti-drift/experiments/run_experiment.py \
    --mode baseline --scene beach --num_chunks 4 \
    > /data/fangziwei/anti-drift/outputs/log_baseline_beach.txt 2>&1 &
PID2=$!

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=29502 \
    /data/fangziwei/anti-drift/experiments/run_experiment.py \
    --mode baseline --scene mountain --num_chunks 4 \
    > /data/fangziwei/anti-drift/outputs/log_baseline_mountain.txt 2>&1 &
PID3=$!

echo "Launched: garden(GPU0:$PID1) beach(GPU2:$PID2) mountain(GPU3:$PID3)"
echo "Logs in /data/fangziwei/anti-drift/outputs/log_baseline_*.txt"

wait $PID1 $PID2 $PID3
echo "All baseline experiments completed!"
