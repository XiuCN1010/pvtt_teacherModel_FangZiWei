# 实验报告: Anti-Drift 后验优化方案 — Baseline 阶段

**日期**: 2026-03-29
**实验环境**: Capstone 服务器 (RTX 5090 ×8, 32GB/卡)
**使用 GPU**: 0, 2, 3, 5 (4 卡并行推理)
**基础模型**: WorldPlay-5B (WAN pipeline, Wan2.2-TI2V-5B-Diffusers)

---

## 1. 做了什么？

### 1.1 总体目标

在 Tencent HY-World 1.5 (WorldPlay-5B) 上建立 anti-drift 后验优化的实验框架。本次是**第一阶段 baseline 实验**：生成 3 个场景的视频，用 DINO 和 PSNR 指标量化 visual drift，为后续的优化方案提供对比基准。

### 1.2 具体工作

1. **环境搭建**: 在 Capstone 服务器上克隆 HY-WorldPlay 仓库，创建 conda 环境 (`worldplay`, Python 3.10)，安装 PyTorch 2.12.0+cu128（支持 RTX 5090 的 sm_120 架构），下载 WAN pipeline 模型权重
2. **代码分析**: 阅读 WAN pipeline 核心代码，找到自回归生成的关键位置（chunk-by-chunk 生成循环），确定 reward 注入点
3. **Reward 模块实现**: 编写 `anti_drift_reward.py`，实现 DINO 一致性奖励和 Temporal PSNR 平滑度奖励
4. **Baseline 实验**: 生成 Garden、Beach、Mountain 三个场景的 4-chunk 视频（61帧，1280×704，16fps），计算评测指标

---

## 2. 怎么做的？

### 2.1 服务器环境搭建

```bash
# 在 Capstone 服务器上
ssh Capstone

# 创建工作目录
mkdir -p ~/anti-drift

# 克隆 HY-WorldPlay
cd ~/anti-drift
git clone https://github.com/Tencent-Hunyuan/HY-WorldPlay.git

# 创建 conda 环境
conda create --name worldplay python=3.10 -y
conda activate worldplay
pip install -r HY-WorldPlay/requirements.txt

# 安装支持 RTX 5090 (sm_120) 的 PyTorch
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall

# 安装 SageAttention (WAN pipeline 必需)
pip install sageattention
```

**关键问题与解决**:
- RTX 5090 使用 sm_120 架构，标准 PyTorch (cu124) 不支持。需要安装 cu128 nightly 版本
- HuggingFace 从服务器无法直接访问，使用 `HF_ENDPOINT=https://hf-mirror.com` 镜像
- 40GB 的 distilled model checkpoint 无法直接加载到 GPU（32GB 显存），修改为先加载到 CPU 再搬到 GPU
- 1280×704 分辨率的 VAE decode 在单卡上 OOM，需要 4 卡并行推理

### 2.2 模型下载

```bash
# WAN pipeline action model (约 9.4GB transformer + 40GB distilled model)
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download tencent/HY-WorldPlay \
    --include 'wan_transformer/*' 'wan_distilled_model/*' \
    --local-dir /data/fangziwei/models/HY-WorldPlay

# Wan2.2 base model (diffusers 格式)
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --local-dir /data/fangziwei/models/Wan2.2-TI2V-5B-Diffusers
```

### 2.3 代码结构与 Reward 注入点分析

**WAN Pipeline 自回归生成流程**:

```
WanRunner.predict()
  ├── 解析 pose → viewmats, Ks, action
  ├── for chunk_i in range(num_chunk):        ← 逐 chunk 生成
  │   ├── pipe.__call__()                     ← 核心去噪循环
  │   │   ├── chunk_i == 0: 初始化 prompt/latents/KV cache
  │   │   ├── chunk_i > 0: 使用历史 latents 作为 context
  │   │   │   ├── select_mem_frames_wan()     ← 选择记忆帧
  │   │   │   ├── KV cache 构建
  │   │   │   └── 4 步去噪 (Flow Matching)
  │   │   └── 保存 latents 到 self.ctx
  │   └── pipe.decode_next_latent()           ← VAE 解码每个 latent
  └── 拼接所有帧 → 输出视频
```

**Reward 注入点** (在 `pipe.__call__` 之后、下一个 chunk 开始之前):
- 每个 chunk 生成 4 个 latent，解码后得到 ~16 帧视频
- 可以在此处**评估当前 chunk 质量**，并决定是否重新生成（Best-of-N 策略）
- 也可以在去噪循环内部注入梯度引导（更复杂，留待后续）

### 2.4 Anti-Drift Reward 模块

实现了两个核心 reward 信号：

#### DINO 一致性奖励 (`DINOReward`)

```python
# 使用 DINOv2 ViT-B/14 提取 CLS token 特征
# 计算两种相似度:
#   1. 相邻帧相似度 (adj_sim): 衡量帧间过渡的平滑度
#   2. 锚点相似度 (anchor_sim): 衡量与首帧的长程一致性

reward = 0.6 * adj_sim + 0.4 * anchor_sim
```

**为什么用 DINO 而不是 CLIP?**
- DINO 对同一物体的身份变化非常敏感（不同个体不会混淆）
- CLIP 更关注语义类别（同类不同个体相似度高），不适合衡量 drift
- VBench 的 Subject Consistency 指标就是基于 DINO，直接对标

#### Temporal PSNR 奖励 (`TemporalPSNRReward`)

```python
# 计算相邻帧的 PSNR (Peak Signal-to-Noise Ratio)
# PSNR 越高 = 帧间变化越小 = 过渡越平滑
# 惩罚 chunk 边界处的突变

reward = clip((mean_psnr - 15) / 25, 0, 1)
```

### 2.5 实验运行

```bash
cd ~/anti-drift/HY-WorldPlay
export PYTHONPATH=$(pwd):$(pwd)/wan:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 4 GPU 并行推理，每个场景约 50-190 秒
CUDA_VISIBLE_DEVICES=0,2,3,5 torchrun --nproc_per_node=4 --master_port=29500 \
    experiments/run_experiment.py --mode baseline --scene garden --num_chunks 4
```

---

## 3. Baseline 实验结果

### 3.1 三场景对比

| 场景 | DINO Adjacent Sim | DINO Anchor Sim | DINO Anchor 下降 | PSNR Mean | PSNR Min | Composite |
|------|-------------------|-----------------|-------------------|-----------|----------|-----------|
| **Garden** | 0.9851 | 0.8325 | **-16.8%** | 14.22 dB | 12.19 dB | 0.4620 |
| **Beach** | 0.9821 | 0.9238 | **-7.6%** | 20.47 dB | 13.06 dB | 0.5451 |
| **Mountain** | 0.9989 | 0.9806 | **-1.9%** | 22.55 dB | 16.63 dB | 0.5865 |

### 3.2 与 016 报告的对比

| 场景 | 016 DINO 末帧 | 016 下降 | 本次 Anchor Sim | 本次下降 | 趋势一致？ |
|------|---------------|----------|-----------------|----------|-----------|
| Garden | 0.8835 | -11.7% | 0.8325 | -16.8% | 是 (最严重) |
| Beach | 0.9569 | -4.3% | 0.9238 | -7.6% | 是 (中等) |
| Mountain | 0.9746 | -2.5% | 0.9806 | -1.9% | 是 (最稳定) |

**趋势完全一致**: Garden 退化最严重 > Beach 中等 > Mountain 最稳定

**差异原因**: 本次实验使用的是 distilled model (4步去噪)，016 可能使用了更多去噪步数；且 prompt 略有不同。

### 3.3 关键发现

1. **DINO Anchor Similarity 是最清晰的 drift 信号**: 所有场景都展现出单调下降趋势，Garden 场景在 61 帧内下降了 16.8%
2. **场景复杂度直接影响 drift 速度**: 复杂场景（Garden: 花草、树木、建筑）退化最快，简单场景（Mountain: 雪山、天空）最稳定
3. **PSNR 与 DINO 的相关性**: 低 PSNR（帧间变化大）通常伴随低 DINO anchor sim（长程一致性差）
4. **Chunk 边界效应**: PSNR min 显著低于 mean，说明 chunk 边界处存在明显的不连续跳变

---

## 4. 这样做的目的

### 4.1 为什么要做这个实验？

在自回归视频世界模型中，**visual drift**（视觉漂移）是一个核心问题。模型每次预测下一个 chunk 时，都是基于自己之前生成的帧作为上下文，而不是真实帧。这导致微小的误差会随时间累积，最终使得生成的视频在视觉质量、场景一致性上逐渐退化。

当前的 anti-drift 方案（如 GameNGen 的 Noise Augmentation、WorldCompass 的 RL 后训练）都没有使用 **DINO 语义一致性**作为直接的优化信号。我们的研究方向是：**用 DINO 特征的帧间余弦相似度作为 RL reward，直接优化模型的长程视觉一致性**。

### 4.2 本次实验在整体研究中的位置

```
整体研究路线:
├── [已完成] 016: Drift 量化分析 — 证明 drift 存在且可量化
├── [已完成] Proposal: 方向设计 — 提出 Learned Noise Aug + DINO RL
├── [本次] Baseline 实验 — 建立评测框架和对比基准    ← 你在这里
├── [下一步] Best-of-N 后验优化 — 验证 DINO reward 的有效性
├── [未来] RL 微调 — 用 DINO reward 做真正的模型训练
└── [未来] Learned Noise Augmentation — 核心创新
```

### 4.3 下一步计划

1. **Best-of-N 后验优化**: 对每个 chunk 生成 3 个候选，用 DINO+PSNR 综合评分选最佳 → 验证 reward signal 是否有效指导生成质量
2. **长序列测试**: 增加到 8/16 chunks，观察 drift 曲线在更长时间上的行为
3. **RL 微调**: 将 DINO reward 接入 WorldCompass 的 DiffusionNFT 训练框架

---

## 5. 代码文件说明

```
anti-drift-experiment/
├── code/
│   ├── anti_drift_reward.py    — Anti-drift reward 模块
│   │   ├── DINOReward          — DINO 一致性奖励 (核心 anti-drift 信号)
│   │   ├── TemporalPSNRReward  — 帧间 PSNR 平滑度奖励
│   │   └── CompositeReward     — 多奖励综合评分器
│   ├── run_experiment.py       — 主实验脚本 (baseline + optimized 模式)
│   └── run_baseline.sh         — 启动脚本 (GPU 分配)
├── results/
│   ├── baseline_garden_metrics.json
│   ├── baseline_beach_metrics.json
│   └── baseline_mountain_metrics.json
└── Experiment_Report_Anti_Drift_Baseline.md  — 本报告
```

**服务器路径**:
- 代码: `/data/fangziwei/anti-drift/experiments/`
- 输出视频: `/data/fangziwei/anti-drift/outputs/`
- HY-WorldPlay 仓库: `/data/fangziwei/anti-drift/HY-WorldPlay/`
- 模型权重: `/data/fangziwei/models/HY-WorldPlay/` 和 `/data/fangziwei/models/Wan2.2-TI2V-5B-Diffusers/`

---

## 6. 关键术语解释

| 术语 | 含义 |
|------|------|
| **Chunk** | 自回归生成的基本单位，每个 chunk 包含 4 个 latent，解码后约 16 帧 |
| **Visual Drift** | 长序列生成中视觉质量逐渐退化的现象，源于误差累积 |
| **DINO Adjacent Sim** | 相邻帧之间的 DINO 特征余弦相似度，衡量短期一致性 |
| **DINO Anchor Sim** | 每帧与首帧的 DINO 特征余弦相似度，衡量长程一致性 (越低 = drift 越严重) |
| **Temporal PSNR** | 相邻帧的峰值信噪比，衡量帧间过渡平滑度 (越低 = 越突变) |
| **Best-of-N** | 后验优化策略: 生成 N 个候选，用 reward 评分选最佳 |
| **Flow Matching** | Wan2.1/HunyuanVideo 使用的扩散模型训练框架，比 DDPM 更高效 |
| **KV Cache** | 将历史帧的 Key-Value 矩阵缓存，新帧通过注意力机制访问历史信息 |
