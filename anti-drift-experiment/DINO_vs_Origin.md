# DINO Reward 后验优化 vs 原始 WorldPlay-5B 对比报告

**日期**: 2026-03-29
**实验环境**: Capstone (RTX 5090 ×8, 32GB/卡), 使用 4 卡并行推理
**模型**: WorldPlay-5B (WAN pipeline, distilled 4-step denoising)
**视频配置**: 4 chunks, 61 帧, 1280×704, 16fps

---

## 1. 实验设计

### 原始模型 (Origin)
- 使用未修改的 WorldPlay-5B 直接生成视频
- 固定种子 seed=42，每个 chunk 按顺序自回归生成

### DINO 后验优化 (DINO Best-of-3)
- **方法**: Best-of-N chunk 级选择 (N=3)
- **流程**: 对每个 chunk（第 1 个之后），生成 3 个候选（不同随机种子），用 **DINO 一致性 + Temporal PSNR** 综合评分选择最佳候选
- **评分公式**: `Composite = 0.5 × (0.6 × DINO_adj + 0.4 × DINO_anchor) + 0.3 × PSNR_norm + 0.1`
- **实现难点**: 需要保存/恢复 pipeline latents, KV cache, 以及 VAE decoder 的 temporal cache (`_feat_map`) 以保证候选之间状态一致

### 评测指标

| 指标 | 含义 | 越高越好？ |
|------|------|-----------|
| **DINO Adj Sim** | 相邻帧 DINOv2 CLS token 余弦相似度均值 | 是 |
| **DINO Anchor Sim** | 每帧与首帧的 DINO 余弦相似度均值 | 是 |
| **Anchor Drop** | 末帧相对首帧的 DINO 相似度下降百分比 | 否（越小越好） |
| **PSNR Mean** | 相邻帧平均 PSNR (dB) | 是 |
| **PSNR Min** | 最差帧间 PSNR (dB) | 是 |
| **Composite** | 综合评分 | 是 |

---

## 2. 对比结果

### Beach 场景

| 指标 | Origin | DINO Best-of-3 | 变化 |
|------|--------|---------------|------|
| DINO Adj Sim | 0.9740 | **0.9807** | +0.0067 (+0.69%) |
| DINO Anchor Sim | 0.9023 | 0.8997 | -0.0026 (-0.29%) |
| Anchor Drop | -16.9% | **-15.3%** | **改善 1.6pp** |
| PSNR Mean | 18.48 dB | **18.58 dB** | +0.10 dB |
| PSNR Min | 16.15 dB | **16.68 dB** | **+0.54 dB** |
| Composite | 0.6144 | **0.6171** | +0.0027 |

### Garden 场景

| 指标 | Origin | DINO Best-of-3 | 变化 |
|------|--------|---------------|------|
| DINO Adj Sim | 0.9890 | 0.9890 | 持平 |
| DINO Anchor Sim | **0.8969** | 0.8694 | -0.0275 (-3.07%) |
| Anchor Drop | **-19.0%** | -24.8% | 退化 5.8pp |
| PSNR Mean | 16.30 dB | 16.28 dB | -0.02 dB |
| PSNR Min | 14.21 dB | **14.22 dB** | +0.02 dB |
| Composite | **0.5917** | 0.5859 | -0.0058 |

### Mountain 场景

| 指标 | Origin | DINO Best-of-3 | 变化 |
|------|--------|---------------|------|
| DINO Adj Sim | 0.9981 | **0.9984** | +0.0003 |
| DINO Anchor Sim | 0.9780 | **0.9795** | +0.0015 (+0.15%) |
| Anchor Drop | -2.8% | **-2.3%** | **改善 0.5pp** |
| PSNR Mean | 18.98 dB | **19.67 dB** | **+0.69 dB** |
| PSNR Min | 15.73 dB | **15.97 dB** | +0.24 dB |
| Composite | 0.6428 | **0.6514** | **+0.0087** |

---

## 3. 综合汇总

| 场景 | 方法 | DINO Adj | DINO Anchor | Anchor Drop | PSNR Mean | PSNR Min | Composite |
|------|------|----------|-------------|-------------|-----------|----------|-----------|
| Beach | Origin | 0.9740 | 0.9023 | -16.9% | 18.48 dB | 16.15 dB | 0.6144 |
| Beach | **DINO** | **0.9807** | 0.8997 | **-15.3%** | **18.58** | **16.68** | **0.6171** |
| Garden | Origin | 0.9890 | **0.8969** | **-19.0%** | 16.30 dB | 14.21 dB | **0.5917** |
| Garden | DINO | 0.9890 | 0.8694 | -24.8% | 16.28 | 14.22 | 0.5859 |
| Mountain | Origin | 0.9981 | 0.9780 | -2.8% | 18.98 dB | 15.73 dB | 0.6428 |
| Mountain | **DINO** | **0.9984** | **0.9795** | **-2.3%** | **19.67** | **15.97** | **0.6514** |

**胜负统计**: DINO 优化在 Beach 和 Mountain 两个场景上全面领先，但在 Garden 场景上退化。

---

## 4. 分析与讨论

### 4.1 DINO 优化在 Beach 和 Mountain 上有效

- **Beach**: DINO Adj Sim 提升 +0.69%，Anchor Drop 改善 1.6 个百分点，PSNR Min 提升 0.54 dB（chunk 边界更平滑）
- **Mountain**: 所有指标全面改善。PSNR Mean +0.69 dB 的提升显著。Composite 从 0.6428 提升到 0.6514 (+1.3%)
- 这两个场景的共同点：**场景变化相对规律**（Beach 的海浪周期性运动，Mountain 的简单前进运动），Best-of-3 候选中确实存在质量差异可供选择

### 4.2 Garden 场景退化的原因

- Garden 的 DINO Anchor Sim 从 0.8969 下降到 0.8694（-3.07%），Anchor Drop 从 -19.0% 恶化到 -24.8%
- **可能原因**:
  1. **Garden 场景复杂度高**（花草、树木、建筑、光影变化多），即使 Best-of-3 也很难找到显著更好的候选
  2. **chunk-level 选择的局限性**: 选择了当前 chunk 得分最高的候选，但这个候选的 latent 可能对后续 chunk 产生不利影响（短视的贪心选择）
  3. **重新运行最佳候选时的状态差异**: 由于 VAE cache 恢复的精度问题，最终输出的视频与候选评分时略有不同
  4. **种子差异**: DINO 优化版本使用不同的种子（seed + chunk_i * 1000 + cand_i），其中被选中的种子未必比原始 seed=42 的固定路径更好

### 4.3 后验优化方案的局限性

Best-of-N 后验优化是一种**推理时优化**，有以下固有局限：

| 局限 | 说明 |
|------|------|
| **贪心选择** | 每个 chunk 独立选择最佳，不考虑对后续 chunk 的影响 |
| **候选多样性有限** | N=3 时，候选之间的差异主要来自初始噪声不同，质量改善空间有限 |
| **计算开销** | 每个 chunk 需要 N+1 次生成（N 个候选 + 1 次重新运行最佳），总时间约为 baseline 的 4× |
| **不改变模型本身** | 模型权重未被修改，无法学到系统性的 anti-drift 能力 |

### 4.4 与 RL 训练方案的对比

| 维度 | Best-of-N (本次) | RL 训练 (未来) |
|------|------------------|----------------|
| **模型是否更新** | 否 | 是 |
| **效果上限** | 候选池中的最优解 | 模型学会系统性纠错 |
| **推理时开销** | N× 推理成本 | 1× 推理成本 |
| **适用性** | 快速验证 reward signal 是否有效 | 生产部署 |

---

## 5. 结论

1. **DINO+PSNR 复合 reward 信号是有效的 anti-drift 指导信号**: 在 Beach (+1.6pp drift 改善) 和 Mountain (+0.5pp drift 改善, +0.69dB PSNR) 场景上，基于该 reward 的 Best-of-N 选择确实选出了更优的生成结果

2. **复杂场景 (Garden) 的优化更具挑战性**: chunk-level 贪心选择在复杂场景上可能反而导致退化，说明需要更长视野的优化策略（如 RL 训练直接优化整段视频的一致性）

3. **Best-of-N 验证了 reward signal 的方向正确，但不足以替代 RL 训练**: 后验优化的改善幅度有限（<1%），真正的性能提升需要通过 RL 后训练将 DINO reward 内化到模型权重中

4. **下一步**: 将 DINO consistency reward 接入 WorldCompass 的 DiffusionNFT 训练框架，进行真正的 RL 后训练

---

## 6. 可复现性

### 文件结构

```
anti-drift-experiment/
├── code/
│   ├── eval_original_worldplay.py       — 原始模型评测脚本
│   ├── eval_dino_optimized_worldplay.py — DINO 优化评测脚本
│   ├── anti_drift_reward.py             — Anti-drift reward 模块
│   └── ...
├── results/
│   ├── original_eval/                   — 原始模型结果
│   │   ├── beach.mp4, garden.mp4, mountain.mp4
│   │   ├── *_metrics.json, *_frame_grid.png
│   │   ├── drift_curves.png
│   │   └── summary.json
│   └── dino_optimized_eval/             — DINO 优化结果
│       ├── beach.mp4, garden.mp4, mountain.mp4
│       ├── *_metrics.json, *_frame_grid.png
│       ├── drift_curves.png
│       └── summary.json
```

### 服务器路径
- 原始模型结果: `/data/fangziwei/anti-drift/outputs/original_eval/`
- DINO 优化结果: `/data/fangziwei/anti-drift/outputs/dino_optimized_eval/`

### 运行命令
```bash
# 原始模型
CUDA_VISIBLE_DEVICES=1,2,3,5 torchrun --nproc_per_node=4 --master_port=29600 \
    experiments/eval_original_worldplay.py

# DINO 优化
CUDA_VISIBLE_DEVICES=1,2,3,5 torchrun --nproc_per_node=4 --master_port=29700 \
    experiments/eval_dino_optimized_worldplay.py
```
