# Proposal: Learned Adaptive Noise Augmentation with DINO-Guided RL for Drift-Resilient Interactive World Models

> **项目**: IP-2026-Spring
> **日期**: 2026-03-23（更新）
> **方向**: 交互式世界模型 / Anti-Drift / RL Post-Training / Noise Augmentation

---

## 目录

1. [背景知识：新人入门指南](#1-背景知识新人入门指南)
2. [HunyuanWorld 1.5 架构详解](#2-hunyuanworld-15-架构详解)
3. [Visual Drift 问题与现有方案](#3-visual-drift-问题与现有方案)
4. [Noise Augmentation 在交互式世界模型中的应用现状](#4-noise-augmentation-在交互式世界模型中的应用现状)
5. [DINO Similarity 作为 Anti-Drift Reward](#5-dino-similarity-作为-anti-drift-reward)
6. [核心研究方向：可学习噪声增强 + DINO RL 奖励](#6-核心研究方向可学习噪声增强--dino-rl-奖励)
7. [实验方案 Proposal](#7-实验方案-proposal)
8. [参考文献](#8-参考文献)

---

## 1. 背景知识：新人入门指南

### 1.1 扩散模型 (Diffusion Models) 基础

扩散模型是当前主流的生成式 AI 框架，用于图像、视频、3D 内容生成。

**核心思想**:
- **前向过程 (Forward Process)**: 向干净数据 `x_0` 逐步添加高斯噪声，经过 `T` 步后变为纯噪声 `x_T`
- **反向过程 (Reverse Process)**: 模型学习从噪声中逐步去噪，最终生成干净数据
- **训练目标**: 预测每一步添加的噪声（DDPM）或预测速度场（Flow Matching，如 Wan2.1 和 HunyuanVideo 使用的框架）

**关键变体**:
- **DDPM**: 最经典的扩散模型形式（Ho et al., 2020）
- **Flow Matching**: 更高效的训练框架，使用线性路径连接噪声和数据（Lipman et al., 2022）。HunyuanVideo、Wan2.1 均基于此。
- **Latent Diffusion Models (LDM)**: 在 VAE 编码的潜在空间中运行扩散过程，大幅降低计算量。Stable Diffusion、HunyuanVideo 均属于此类。

**视频扩散模型**:
视频生成在图像扩散模型基础上增加时间维度。输入形状从 `[B, C, H, W]` 扩展为 `[B, C, T, H, W]`（T为帧数）。关键挑战在于时间一致性——相邻帧之间需要在外观、运动上保持连贯。

---

### 1.2 Transformer 在生成模型中的应用 (DiT)

**DiT (Diffusion Transformer)**:
将 UNet 替换为 Transformer 作为扩散模型的骨干网络（Peebles & Xie, 2023）。现代视频生成模型（HunyuanVideo、Wan2.1）均基于 DiT 架构。

**Self-Attention 类型**（对理解时间一致性至关重要）:
| 注意力类型 | 作用范围 | 对 Drift 的意义 |
|---|---|---|
| Spatial Self-Attention | 单帧内 patch 间 | 减少帧内闪烁 |
| Temporal Self-Attention | 相邻帧对应位置 | 核心时序一致性机制 |
| Spatiotemporal Self-Attention | 所有帧全局相关 | 最强一致性，计算开销大 |
| Causal Self-Attention | 当前帧 + 历史帧 | 自回归生成的标准设置 |

---

### 1.3 自回归视频生成 (Autoregressive Video Generation)

**概念**: 模型每次生成固定长度的视频片段（chunk），将已生成的视频作为上下文（context），预测下一个 chunk。类似于语言模型的 next-token prediction，这里是 next-chunk prediction。

**Exposure Bias（暴露偏差）问题**:
训练时模型看到干净的真实帧作为上下文；推理时却要接收自己生成的（可能有瑕疵的）帧。这种训练/推理分布不一致导致错误随时间累积，即 **Visual Drift**。

**KV Cache 在视频生成中的应用**:
将历史帧的 Key-Value 矩阵缓存，新帧生成时通过交叉注意力访问历史信息，避免重复计算。这是 Rolling Forcing、Self-Forcing 等方法的基础机制。

---

### 1.4 强化学习微调生成模型 (RL Fine-Tuning)

这是本实验的核心方法框架。

**基本流程**:
1. 使用预训练生成模型作为 **Policy**（策略）
2. 生成视频样本（rollout）
3. 用 **Reward Function** 评分（越高越好）
4. 用 Policy Gradient 方法更新模型参数，最大化期望奖励

**主要 RL 算法变体**:
- **RLHF (Reinforcement Learning from Human Feedback)**: 用人类偏好数据训练奖励模型，再用 PPO 优化。InstructGPT 等 LLM 对齐的基础方法。
- **GRPO**: Group Relative Policy Optimization，相对于 PPO 更简单，通过组内相对排名计算优势函数，被 DeepSeek-R1 等模型广泛使用。
- **DiffusionNFT (Negative-aware Fine-Tuning)**: 专为扩散模型设计的 RL 算法，基于 Flow Matching。HunyuanWorld 1.5 的 WorldCompass 使用此算法。
- **Re-DMD (Reward-Weighted Distribution Matching Distillation)**: 将奖励分数作为权重融入分布匹配蒸馏损失，避免通过奖励模型反向传播。

**关键概念 — Reward Hacking（奖励欺骗）**:
模型找到可以最大化奖励函数但违背实际目标的"捷径"。例如：若只优化视觉一致性奖励，模型可能生成静态视频（每帧相同）来获得最高分。解决方案：多奖励组合（如 WorldCompass 的 IF + VQ 双奖励设计）。

---

### 1.5 DINO / DINOv2 基础

**DINO (Self-**DI**stillation with **NO** labels)**:
Facebook Research 提出的自监督视觉表示学习方法（Caron et al., 2021）。使用 Vision Transformer (ViT) 作为骨干，通过自蒸馏（student-teacher 框架）在无标签数据上学习丰富的视觉特征。

**DINOv2**（Oquab et al., 2023）: 改进版本，更大规模数据（LVD-142M）和更好的特征质量。常用变体：
- `ViT-B/14`: 86M 参数，patch size 14×14（标准选择）
- `ViT-L/14`: 307M 参数，更强特征质量
- `ViT-g/14`: 1.1B 参数，最强但最重

**DINO 特征的独特性质**:
1. **语义敏感**: 对同一物体在不同姿态、光照下的特征保持相似；对物体身份（identity）变化非常敏感
2. **非类别聚合**: 不会把同类别不同个体混为一谈（与 CLIP 的区别），更适合衡量同一视频中主体的一致性
3. **Patch-level 特征**: 输出 `N_patch × D` 的特征图（不只是全局 CLS token），可以进行像素级对应

**DINO Cosine Similarity 计算**:
```python
import torch
import torch.nn.functional as F

# 提取 DINOv2 特征（使用 CLS token 或平均 patch features）
feat_t1 = dino_model(frame_t1)  # [B, D]
feat_t2 = dino_model(frame_t2)  # [B, D]

# Cosine Similarity
similarity = F.cosine_similarity(feat_t1, feat_t2, dim=-1)  # [B], 范围 [-1, 1]
```

---

### 1.6 VBench 评测体系

VBench 是视频生成模型的综合评测基准（CVPR 2024）。本实验关注的核心指标：

| 维度 | 度量方式 | 意义 |
|---|---|---|
| **Subject Consistency** | DINO ViT 特征余弦相似度（帧间） | 主体外观一致性，即 Anti-Drift 的主要评测指标 |
| **Background Consistency** | CLIP 特征余弦相似度（帧间） | 背景一致性 |
| **Motion Smoothness** | AMFlow 光流估计的平滑度 | 运动连贯性 |
| **Dynamic Degree** | RAFT 光流幅度 | 视频是否有实质运动（防止生成静态视频） |
| **Aesthetic Quality** | LAION-AI Aesthetic Predictor | 视觉美观度 |

**重要**: 若你的 Anti-Drift Reward 使用 DINO Similarity，那么 VBench Subject Consistency 就是最直接的评测对标指标。

---

## 2. HunyuanWorld 1.5 架构详解

HunyuanWorld 系列包含两个技术路线：

### 2.1 HunyuanWorld 1.0：离线沉浸式 3D 世界生成

**论文**: HunyuanWorld 1.0: Generating Immersive, Explorable, and Interactive 3D Worlds from Words or Pixels (arxiv: 2507.21809)

这是一个从文本或单图像生成可探索三维世界的多阶段 Pipeline，并非实时流式系统。

**核心组件**:

```
输入（文本/图像）
    ↓
┌─────────────────────────────────────────┐
│ 1. LLM 增强提示词处理                    │
│   - 将简短描述扩展为结构化场景描述         │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 2. Panorama-DiT（360° 全景生成）          │
│   - 基于 DiT 框架生成等矩形全景图          │
│   - 处理边界连续性（环形去噪 + 渐进混合）   │
│   - 支持文本条件和图像条件两种模式         │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 3. Agentic World Layering（场景分层）     │
│   - VLM 驱动的前景/背景语义分解           │
│   - Grounding DINO 在全景图上做目标检测   │
│   - NMS 处理跨边界重叠检测框              │
│   - 输出：分离的语义层（前景对象、背景、天空）│
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 4. Layer-Wise 3D Reconstruction（分层重建）│
│   - 前景对象：Hunyuan3D 重建为完整 3D Mesh│
│   - 背景：深度对齐的片状变形 (Sheet Warp) │
│   - 天空：均匀深度 Mesh 或 HDRI 环境贴图  │
│   - 跨层深度对齐：最小化层间几何距离       │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 5. World Extension (Voyager)             │
│   - 3D 点云世界缓存提供空间上下文          │
│   - 扩散引导的自回归视频扩展               │
│   - 实现长程世界漫游                      │
└─────────────────────────────────────────┘
```

**技术亮点**:
- 语义分层 Mesh 表示：结构化、可交互的 3D 场景
- 跨层深度对齐：保证几何一致性
- 层补全网络：处理遮挡区域的全景修复

---

### 2.2 HY-World 1.5 / WorldPlay：实时交互世界模型

**技术报告**: HY-World 1.5 Tech Report (Dec 2025)
**GitHub**: https://github.com/Tencent-Hunyuan/HY-WorldPlay

这是**本实验最相关的系统**——一个基于视频扩散的实时交互式世界模型，用户通过键鼠输入控制，模型实时生成游戏画面。

**系统架构概览**:

```
┌──────────────────────────────────────────────────┐
│                HY-World 1.5 总体架构               │
│                                                  │
│  用户输入（键盘+鼠标）                              │
│       ↓                                          │
│  Dual Action Representation（双动作表示）           │
│       ↓                                          │
│  [历史 Chunk 1] [历史 Chunk 2] ... [目标位置]       │
│       ↓                                          │
│  Reconstituted Context Memory（重构上下文记忆）      │
│       ↓                                          │
│  ┌──────────────────┐   ┌──────────────────────┐ │
│  │ HunyuanVideo     │ 或 │    WAN Pipeline      │ │
│  │ Pipeline (8B)    │   │       (5B)           │ │
│  │ 更强动作控制      │   │  更低显存需求         │ │
│  └──────────────────┘   └──────────────────────┘ │
│       ↓                                          │
│  Four-Step Distillation（四步蒸馏）                 │
│       ↓                                          │
│  实时输出 @24 FPS                                  │
└──────────────────────────────────────────────────┘
```

**核心组件详解**:

#### 组件 1: Dual Action Representation（双动作表示）
- **功能**: 将用户的键盘输入和鼠标输入分别编码为双路 Action Token
- **重要性**: 相比单路编码，双路分离表示更好地捕捉键盘（离散按键）和鼠标（连续移动/点击）的异质性，提升动作执行精度
- **实现**: Action Token 通过 Cross-Attention 注入到 DiT 的每一层

#### 组件 2: Reconstituted Context Memory（重构上下文记忆）
- **功能**: 动态重建历史帧上下文，在滑动窗口限制下保留几何重要性高的远期帧
- **解决的问题**: 自回归生成中的"记忆衰减"——标准滑动窗口只保留最近 N 帧，导致模型遗忘早期重要场景（如出现过的关键地标）
- **机制**: 重要性评分筛选历史帧 → 选择性保留远期关键帧 → 与近期帧混合形成重构上下文

#### 组件 3: Context Forcing（上下文强制对齐）
- **功能**: 对长序列自回归生成的上下文记忆进行蒸馏对齐，保持 teacher 模型和 student 模型的上下文表示一致
- **解决的问题**: 蒸馏压缩过程中可能丢失长程信息
- **机制**: 在蒸馏训练时约束 student 的 context 表示与 teacher 对齐（通过 KL 散度或 MSE loss）

#### 组件 4: Four-Step Distillation（四步蒸馏）
- **功能**: 将标准扩散采样的多步（通常 20-50 步）压缩到 4 步，实现 24 FPS 实时生成
- **解决的问题**: 扩散模型天然的多步采样导致高延迟
- **机制**: 基于分布匹配蒸馏（Distribution Matching Distillation），将 teacher 的多步分布压缩到 student 的 4 步分布

---

### 2.3 WorldCompass：RL 后训练框架

WorldCompass 是 HY-World 1.5 配套的 RL 后训练系统（arxiv: 2602.09022），专门解决长程交互世界模型中的动作执行准确性问题。

**RL 算法**: DiffusionNFT（基于 Flow Matching 的负感知微调）
- 无 KL 散度正则化（依赖低学习率 + EMA 更新维持稳定性）
- 优势函数: `a_j = (s_j - mean) / std`
- 优化概率: `r = 0.5 + 0.5 × clip[λ·a_IF + (1-λ)·a_VQ, -1, 1]`

**奖励设计（双奖励互补）**:

| 奖励类型 | 度量内容 | 实现方式 |
|---|---|---|
| **Interaction Following (IF)** | 动作执行准确度（相机轨迹） | WorldMirror + DepthAnythingV3 估计相机位姿，对比旋转/平移精度 |
| **Visual Quality (VQ)** | 视觉美学和文本对齐 | HPSv3 (Human Preference Score v3) |

**关键设计决策 — Clip-Level Rollout**:
- 在单个目标位置采样 G 个候选 clip，前缀共享
- 将复杂度从 O(N·G) 降低到 O(N+G)
- 强制模型依赖自身不完美预测（缓解 Exposure Bias）

**训练结果**:
- 动作执行准确率: 20% → 55%（长序列 381 帧）
- HPSv3 视觉质量: +1.82
- 硬件: 64 × H20 GPU，3 天

---

## 3. Visual Drift 问题与现有方案

### 3.1 问题定义

**Visual Drift（视觉漂移）**: 在长序列自回归视频生成中，随着时间步增加，生成视频的视觉质量和内容一致性逐渐下降的现象。

**成因分类**:

```
Visual Drift 的根因
├── Exposure Bias（暴露偏差）
│   └── 训练时输入为真实帧，推理时输入为模型自生成帧
│       → 训练/推理分布不匹配 → 误差累积
├── Error Accumulation（误差累积）
│   └── 每帧的微小误差在下一帧中被当作"真实"上下文
│       → 误差随时间指数级放大
└── Reward Hacking（奖励欺骗）
    └── 在 RL 训练中，过度优化单一奖励会损害其他质量维度
```

### 3.2 主流解决方案概览

#### 类别 A：训练时分布对齐（Training-Time Approaches）

| 方案 | 核心思想 | 可行性 | 关键论文 |
|---|---|---|---|
| **Noise Augmentation** | 训练时对历史帧加随机噪声，教模型自我纠错 | ★★★★★ 极高 | GameNGen (2024) |
| **BAgger** | 将生成的漂移序列时间逆转作为纠正轨迹训练 | ★★★★☆ 高 | arxiv 2512.12080 |
| **Stable Video Infinity** | Error Recycling：将误差存入 Replay Memory 重采样到未来 mini-batch | ★★★★☆ 高 | ICLR 2026 Oral |
| **Self-Forcing** | 直接在自回归 rollout 上训练（含 KV Cache），模拟推理 | ★★★★☆ 高 | NeurIPS 2025 Spotlight |
| **Resampling Forcing** | 无 teacher 的自重采样方案，适合从零训练 | ★★★★☆ 高 | 2025 |

**推荐入门方案**: **Noise Augmentation**（最简单），或 **BAgger**（无需 teacher 网络、不需 BPTT）

#### 类别 B：推理时架构优化

| 方案 | 核心思想 | 可行性 |
|---|---|---|
| **Rolling Forcing** | 多帧联合去噪 + 双向注意力 + Attention Sink | ★★★★☆ 高 |
| **Reconstituted Context Memory** | 选择性保留重要历史帧 | ★★★★☆ 高（HY-World 1.5 已实现）|
| **RefDrop** | 将参考图像融入 UNet 自注意力 | ★★★☆☆ 中（单参考，长视频受限）|

#### 类别 C：基于 RL 奖励的后训练

| 方案 | 核心思想 | 可行性 |
|---|---|---|
| **WorldCompass (HY-World 1.5)** | IF + VQ 双奖励的 DiffusionNFT | ★★★★☆ 高 |
| **RLVR-World** | L1 + LPIPS 与参考帧的距离作为奖励 | ★★★★☆ 高（需参考帧）|
| **Reward Forcing (Re-DMD)** | 奖励加权分布匹配蒸馏 | ★★★★☆ 高 |
| **DINO Reward（本实验）** | DINO 帧间余弦相似度作为奖励 | ★★★★☆ 高（潜力大）|
| **PRFL** | 用预训练视频模型本身作为潜在空间奖励模型 | ★★★★☆ 高 |

### 3.3 方案可行性深度评估

**最具综合潜力的组合**（参考 WorldCompass 的多奖励设计）:

```
Anti-Drift Composite Reward =
    λ₁ × DINO_Subject_Consistency     ← 主体外观一致性（本实验核心）
  + λ₂ × Motion_Quality_Score         ← 防止静态视频（RAFT 或 VLM based）
  + λ₃ × Aesthetic_Score (optional)   ← 防止质量退化（HPSv3 或 LAION AE）
```

---

## 4. Noise Augmentation 在交互式世界模型中的应用现状

### 4.1 GameNGen：原始方案（ICLR 2025）

**论文**: Diffusion Models Are Real-Time Game Engines (arxiv: 2408.14837)

GameNGen 是 Noise Augmentation 用于交互式世界模型纠错的开创性工作。

**技术细节**:
- 架构: UNet (Stable Diffusion 1.4)
- 训练阶段: RL Agent 玩 DOOM 收集游戏数据 → 训练扩散模型预测下一帧
- **Noise Augmentation 机制**: 训练时对 context frames 的 latent 编码加高斯噪声
  - 噪声强度 α 均匀采样自 [0, 0.7]
  - 离散化为 10 个 bucket，bucket index 作为额外条件输入
  - 推理时 α=0，但模型已学会从含噪上下文中恢复
- **效果**: 无 Noise Aug 时 LPIPS 在 10-20 步后快速退化；有 Noise Aug 后数千帧保持稳定
- **局限**: UNet 架构、单游戏（DOOM）、固定各向同性高斯噪声、未与其他 anti-drift 方法对比

### 4.2 各交互式世界模型的 Anti-Drift 方案对比

#### 使用了 Noise Augmentation（或变体）的模型

| 模型 | 架构 | Noise Aug 方式 | 与 GameNGen 的区别 |
|---|---|---|---|
| **GameNGen** (ICLR 2025) | UNet (SD 1.4) | Context frames 加均匀高斯噪声，10 bucket 条件化 | 原始方案 |
| **GameGen-X** (ICLR 2025) | DiT (MSDiT) | 仅对 initial frames 加轻微高斯噪声 | 更简单，未引用 GameNGen |
| **Oasis** (Decart) | DiT | Dynamic Noising：基于 Diffusion Forcing，推理时按 schedule 注入/移除噪声 | 概念相关，技术路线不同 |
| **HY-World 1.5** | DiT | 受 Diffusion Forcing 启发，不同 chunk 施加不同噪声级别 | Chunk-level，非 frame-level |
| **Vid2World** | DiT | 每帧独立采样噪声级别 k_t ~ U(0,K) | 更接近 Diffusion Forcing 范式 |

#### 不使用 Noise Augmentation 的模型

| 模型 | 架构 | 替代 Anti-Drift 方案 |
|---|---|---|
| **DIAMOND** (NeurIPS 2024) | UNet (EDM) | EDM formulation：高噪声级别下更好的分数估计 |
| **Genie 1** (DeepMind) | ST-Transformer (MaskGIT) | 有限记忆窗口（16帧），Drift 仍是根本局限 |
| **Genie 2/3** (DeepMind) | Latent Diffusion | 扩展记忆/长程上下文（技术细节未公开）|
| **Hunyuan-GameCraft** (Tencent) | DiT | Sink tokens + block-sparse attention + 混合历史条件化 |
| **Matrix-Game 2.0** (Skywork AI) | DiT | **Self-Forcing** + DMD 蒸馏（直接在自回归 rollout 上训练）|
| **GameFactory** (Kuaishou/Kling) | Video Diffusion | 多阶段训练 + 动作解耦 |

### 4.3 关键研究空白（Gap Analysis）

经过全面调研，我们识别出以下 **4 个核心研究空白**：

```
Gap 1: Noise Augmentation + DINO RL Reward 从未组合过
├── GameNGen 的 Noise Aug 仅用重建损失（MSE/L1）训练
├── DINO 奖励在 RL 视频生成中是新兴方向
└── 两者结合 = 完全空白

Gap 2: 没有 Anti-Drift 方法的系统性对比
├── 每篇论文只与自己的消融比较
└── Noise Aug vs Diffusion Forcing vs Self-Forcing vs EDM
    在同一基准上的 head-to-head 对比 = 不存在

Gap 3: 噪声始终是固定高斯，无人尝试可学习/自适应噪声
├── MuLAN (NeurIPS 2024 Spotlight) 证明自适应噪声在图像生成中更优
├── 不同空间区域（天空 vs 移动物体 vs UI）需要不同噪声强度
└── Learned noise 迁移到交互式世界模型 = 空白

Gap 4: GameNGen 方案未在 DiT 架构上系统验证
├── Oasis/HY-World 用了类似思想但技术实现差异大
└── 原始 bucket-conditioned 方案在 DiT 上的效果 = 未知
```

**这 4 个空白共同指向一个有力的可发表方向（见第 6 节）。**

---

## 5. DINO Similarity 作为 Anti-Drift Reward

### 4.1 技术原理

**VBench 中的标准定义**（本实验的评测依据）:

给定视频 `V = {f_1, f_2, ..., f_T}`，Subject Consistency 定义为：

```
SC(V) = (1 / (T-1)) × Σ_{t=2}^{T} cos(φ_DINO(f_t), φ_DINO(f_{t-1}))
```

其中 `φ_DINO` 为 DINOv2 CLS token 特征提取函数。

**作为 RL 奖励信号**:

在 RL 训练中，对每个生成的 chunk（16 帧），计算帧间 DINO 一致性得分，将其作为 Reward：

```python
def dino_consistency_reward(frames, dino_model):
    """
    frames: List of PIL Images or Tensors, shape [T, C, H, W]
    returns: scalar reward in [0, 1]
    """
    features = []
    for frame in frames:
        feat = dino_model.get_cls_token(frame)  # [D]
        features.append(F.normalize(feat, dim=-1))

    similarities = []
    for i in range(1, len(features)):
        sim = torch.dot(features[i], features[i-1]).item()
        similarities.append((sim + 1) / 2)  # 归一化到 [0, 1]

    return sum(similarities) / len(similarities)
```

### 4.2 支撑证据

| 证据来源 | 内容 | 相关性 |
|---|---|---|
| **VBench (CVPR 2024)** | DINO 余弦相似度是视频主体一致性的 Gold Standard 指标 | 直接验证了 DINO 对 Drift 的敏感性 |
| **DINO-WM (ICML 2025)** | DINOv2 特征距离作为世界模型规划目标，实现零样本任务泛化 | 证明 DINO 距离可作为有效优化目标 |
| **Back to Features (2025)** | 直接在 DINOv2 特征空间训练世界模型，大幅降低计算量 | 证明 DINO 特征空间适合视频世界模型 |
| **Rewarding DINO (2026)** | DINOv3 特征用于机器人操作任务奖励建模，Kendall τ=0.82 | 证明 DINO 特征可作为高质量奖励信号 |
| **Perceptual Straightening (2025)** | 真实视频在 DINO 特征空间的轨迹比 AI 生成视频更平滑 | 证明 DINO 平滑度与视觉自然度正相关 |
| **BAgger (2024)** | 使用 VBench Subject Consistency (+3.35) 评测，隐式优化 DINO 一致性 | 说明 DINO 奖励目标可被 RL 有效优化 |

### 4.3 挑战与风险

| 挑战 | 描述 | 缓解措施 |
|---|---|---|
| **Reward Hacking（静态视频）** | 生成每帧相同的近静态视频可获得最高 DINO 相似度 | 加入 Motion Quality 互补奖励（Dynamic Degree / RAFT 光流幅度）|
| **语义 vs. 几何一致性** | DINO 捕捉语义/外观，不擅长几何/深度一致性 | 对几何精度要求高时，可辅以 DepthAnythingV3 相机位姿奖励 |
| **Patch 粒度限制** | ViT-B/14 的 patch size=14px，像素级精度有限 | 考虑 ViT-L/14 或使用 patch-level similarity（非仅 CLS token）|
| **计算开销** | 每次采样都需要 DINOv2 前向推理 | DINOv2 ViT-B 推理速度较快，通常不成为瓶颈 |

---

## 6. 核心研究方向：可学习噪声增强 + DINO RL 奖励

### 6.1 研究问题

> **世界模型的自纠错能力能否通过感知级奖励信号被显式优化？**

当前所有 Noise Augmentation 方案都是"开环"的——训练时加噪声，但优化目标仍是像素级重建损失（MSE/L1）。我们提出将这个过程升级为"闭环"：用 DINO 语义一致性作为 RL 奖励，直接优化长程视觉一致性目标。

### 6.2 三重新颖性分析

```
当前技术现状                          本方案
─────────────────────────────────────────────────────────────────
固定各向同性高斯噪声                →  可学习的空间自适应噪声策略网络 π_θ
  (GameNGen: α ~ U[0, 0.7])            (轻量 CNN，输出空间噪声图)

仅用像素级重建损失训练              →  DINO 语义一致性 RL 奖励闭环优化
  (MSE/L1 loss)                         (直接优化长程 Subject Consistency)

单方法、单游戏验证                  →  首个系统性 Anti-Drift Benchmark
  (每篇论文只做自己的消融)              (5+ 方法在统一基准上 head-to-head 对比)
```

**新颖性 1 — 可学习噪声增强**:
- 不同空间区域对噪声的需求不同：天空（低频，少噪声即可）、移动物体（高频，需要更强增强）、UI 元素（应保持清晰）
- 用轻量 CNN 预测 spatially-varying noise map，条件化于当前游戏状态
- MuLAN (NeurIPS 2024 Spotlight) 已证明自适应噪声在图像扩散模型中优于固定高斯，但**未被迁移到交互式世界模型**

**新颖性 2 — DINO Reward 闭环**:
- 当前 Noise Augmentation 与最终目标（长程视觉一致性）之间是间接关系
- 通过 DDPO/DRaFT 风格 RL，将 DINO 帧间一致性作为奖励信号，直接优化"N 步自回归后的一致性保持度"
- 实质上将 anti-drift 从隐式约束变为显式优化目标

**新颖性 3 — 系统性 Benchmark**:
- 目前不存在 Noise Aug vs Diffusion Forcing vs Self-Forcing vs EDM 的统一对比
- 在 DIAMOND（Atari，开源）和/或 Oasis（Minecraft，开源 500M）上建立标准化评测
- 填补领域关键空白

### 6.3 为什么这不是简单的"A+B"拼凑？

这个方向回答的核心科学问题是：

> **在自回归世界模型中，error correction 的最优策略是由数据驱动学习得到的，还是由人工设计（固定高斯）就足够的？DINO 语义特征能否作为比像素级损失更有效的纠错监督信号？**

如果实验证明可学习噪声 + DINO 奖励显著优于固定高斯 + MSE（预期如此），则说明：
1. Anti-drift 需要语义级而非像素级的监督
2. 噪声增强策略本身是可优化的，GameNGen 的方案只是一个粗糙起点
3. 为未来的世界模型训练提供了更 principled 的 anti-drift 框架

### 6.4 与现有工作的关系定位

```
                    训练时纠错
                        │
            ┌───────────┼───────────┐
            │           │           │
     Noise Aug      Self-Forcing  Diffusion Forcing
     (GameNGen)    (Matrix-Game)   (Oasis/HY-World)
            │
            │  ← 本方案的改进方向
            ↓
    Learned Noise Aug
    + DINO RL Reward
    (Ours)
            │
            ├── vs 固定高斯:      空间自适应 > 各向同性
            ├── vs 像素级损失:    语义级奖励 > MSE/L1
            └── vs 单方法验证:    系统 benchmark > 孤立消融
```

### 6.5 发表目标与定位

| 目标会议 | 定位 | 所需强度 |
|---|---|---|
| **ICLR / NeurIPS / ICML** | 主会 | 需要在 2+ 环境上显著改进，且 benchmark 有影响力 |
| **ECCV / CVPR** | 主会 | 视觉生成社区，更注重生成质量的提升 |
| **AAAI** | 主会 | 方法论贡献 + 充分实验即可 |
| **Workshop (保底)** | NeurIPS/ICML Workshop on World Models | 初步结果即可投稿 |

---

## 7. 实验方案 Proposal

### 7.1 研究目标

**核心问题**: 能否用可学习的自适应噪声增强策略（替代固定高斯噪声），结合 DINOv2 帧间余弦相似度作为 RL Anti-Drift Reward，显著提升交互式世界模型在长序列生成中的视觉一致性？

**预期贡献**:
1. 提出 Learned Adaptive Noise Augmentation：首次将可学习噪声迁移到交互式世界模型
2. 提出 DINO-guided RL 闭环优化：首次用 DINO 语义一致性奖励直接优化 anti-drift
3. 建立首个 Anti-Drift 方法系统性 Benchmark

---

### 7.2 实验框架设计

```
┌──────────────────────────────────────────────────────────────────┐
│                    实验总体框架（更新版）                           │
│                                                                  │
│  Pre-trained World Model                                         │
│  (DIAMOND / Oasis 500M / Wan2.1+DiffSynth)                      │
│           ↓                                                      │
│  ┌──────────────────────────────────────┐                        │
│  │  Noise Augmentation Module           │                        │
│  │                                      │                        │
│  │  Variant A: 固定高斯 (GameNGen 复现)  │                        │
│  │  Variant B: 可学习噪声策略 π_θ (Ours) │ ← 核心创新             │
│  │    - 输入: context frame latent       │                        │
│  │    - 输出: spatial noise map          │                        │
│  └──────────────────────────────────────┘                        │
│           ↓                                                      │
│  ┌──────────────────────────────────────┐                        │
│  │  Autoregressive Rollout (N chunks)   │                        │
│  │  每 chunk = 16 帧                    │                        │
│  └──────────────────────────────────────┘                        │
│           ↓                                                      │
│  ┌──────────────────────────────────────┐                        │
│  │  Composite Reward                    │                        │
│  │                                      │                        │
│  │  R = λ₁ × DINO_Consistency(全序列)   │ ← Anti-Drift 核心      │
│  │    + λ₂ × Motion_Quality             │ ← 防静态视频           │
│  │    + λ₃ × Reconstruction_Quality     │ ← 像素级基线质量        │
│  └──────────────────────────────────────┘                        │
│           ↓                                                      │
│  ┌──────────────────────────────────────┐                        │
│  │  RL Update                           │                        │
│  │  (同时更新 世界模型 + 噪声策略 π_θ)    │                        │
│  │  DDPO / DiffusionNFT / Re-DMD        │                        │
│  └──────────────────────────────────────┘                        │
│           ↓                                                      │
│  Drift-Resilient World Model + Optimized Noise Strategy          │
└──────────────────────────────────────────────────────────────────┘
```

---

### 7.3 噪声策略网络 π_θ 设计

```python
import torch
import torch.nn as nn

class AdaptiveNoisePolicy(nn.Module):
    """
    可学习的空间自适应噪声策略网络。
    输入 context frame 的 latent，输出 spatially-varying noise map。
    """
    def __init__(self, latent_channels=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            # 3层轻量 CNN，保持空间分辨率
            nn.Conv2d(latent_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, latent_channels, 3, padding=1),
            nn.Sigmoid(),  # 输出范围 [0, 1]，乘以 max_alpha 后为噪声强度
        )
        self.max_alpha = nn.Parameter(torch.tensor(0.7))  # 可学习的最大噪声强度

    def forward(self, context_latent):
        """
        context_latent: [B, C, H, W] - VAE 编码后的 context frame
        returns: [B, C, H, W] - 空间噪声强度图 ∈ [0, max_alpha]
        """
        noise_map = self.net(context_latent) * self.max_alpha.clamp(0, 1)
        return noise_map

    def augment(self, context_latent):
        """
        对 context latent 施加自适应噪声增强
        """
        noise_map = self.forward(context_latent)
        noise = torch.randn_like(context_latent)
        augmented = context_latent + noise_map * noise
        return augmented, noise_map
```

**设计理念**:
- 参数量极少（~50K params），不影响训练/推理速度
- Sigmoid + max_alpha clamp 确保噪声强度有界
- GroupNorm + SiLU 是扩散模型中的标准组件
- 推理时可选择：使用学到的 noise map（在线纠错）或设为 0（标准推理）

---

### 7.4 DINO 长程一致性奖励设计

```python
class DINODriftReward:
    """
    改进的 DINO 奖励：不仅衡量相邻帧一致性，还衡量长程衰减速率
    """
    def __init__(self, model_name="facebook/dinov2-base", device="cuda"):
        self.dino = DINORewardModel(model_name, device)

    def compute_reward(self, all_frames, chunk_size=16):
        """
        all_frames: 完整 rollout 的所有帧 [T_total, C, H, W]
        returns: composite reward scalar

        关键改进：不只算相邻帧的相似度，还衡量"衰减曲线的面积"
        """
        features = self.dino.get_features(all_frames)  # [T, D]

        # 1. 相邻帧一致性（标准 VBench Subject Consistency）
        adj_sim = (features[:-1] * features[1:]).sum(dim=-1)  # [T-1]
        adj_reward = ((adj_sim + 1) / 2).mean().item()

        # 2. 与首帧的长程一致性衰减曲线
        anchor = features[0:1]  # [1, D]
        long_sim = (features * anchor).sum(dim=-1)  # [T]
        # AUC: 衰减曲线下面积（越大越好，表示一致性保持越久）
        long_auc = ((long_sim + 1) / 2).mean().item()

        # 3. 衰减速率惩罚（slope 越负，Drift 越严重）
        t = torch.arange(len(long_sim), dtype=torch.float32)
        slope = (t * long_sim).mean() - t.mean() * long_sim.mean()
        slope = slope / (t.var() + 1e-8)
        slope_penalty = slope.clamp(min=-1, max=0).item()  # 只惩罚下降

        # 组合
        return 0.5 * adj_reward + 0.3 * long_auc + 0.2 * (1 + slope_penalty)
```

### 7.5 逐阶段实验计划

#### 阶段 0：环境搭建与基线评测（Week 1-2）

**任务**:
- [ ] 选择基础世界模型：**优先 DIAMOND**（Atari，开源，单卡可跑）或 **Oasis 500M**
- [ ] 复现 GameNGen 式固定高斯 Noise Augmentation 作为 baseline
- [ ] 配置 DINOv2 特征提取（`facebook/dinov2-base`）
- [ ] 建立评测 pipeline：VBench Subject Consistency + Dynamic Degree + LPIPS@{100,500,1K步}
- [ ] 绘制基线模型的 DINO 一致性衰减曲线（帧 → DINO sim with frame 0）

**产出**: 基线 Drift 量化报告，明确改进空间

**关键本地资源**:
- `/e/IP-2026-Spring_Source/DiffSynth-Studio` — DiffSynth Studio 代码
- `/e/IP-2026-Spring_Source/Wan2.1` — Wan2.1 代码

---

#### 阶段 1：固定噪声 + DINO RL Reward（Week 3-4）

**任务**:
- [ ] 实现 RL 微调 Loop（DDPO 或 DiffusionNFT）
- [ ] 使用 GameNGen 式固定高斯噪声 + DINO RL Reward 训练
- [ ] 对比：固定噪声+MSE loss vs 固定噪声+DINO reward
- [ ] 验证 DINO Reward 是否比 MSE 更有效地抑制 Drift

**成功标准**: DINO reward 版本的 Subject Consistency 优于 MSE 版本 ≥ 1.0

**这一阶段回答**: "DINO 语义奖励是否比像素级损失更适合做 anti-drift 监督？"

---

#### 阶段 2：可学习噪声策略（Week 5-7）— 核心实验

**任务**:
- [ ] 实现 AdaptiveNoisePolicy 网络（见 7.3 节代码）
- [ ] 端到端训练：世界模型 + 噪声策略网络联合 RL 优化
- [ ] 消融矩阵：

| 配置 | Noise 类型 | 损失/奖励 | 预期 |
|---|---|---|---|
| A: Baseline | 无 | MSE | Drift 严重 |
| B: GameNGen 复现 | 固定高斯 | MSE | 中等改善 |
| C: 固定高斯 + DINO RL | 固定高斯 | DINO Reward | 较好改善 |
| **D: Learned + DINO RL (Ours)** | 可学习自适应 | DINO Reward | **最佳** |
| E: Diffusion Forcing | Per-frame noise | MSE | 对比方法 |
| F: Self-Forcing | N/A (rollout 训练) | MSE | 对比方法 |

- [ ] 可视化噪声策略网络学到的 noise map（哪些区域被重点增强？）
- [ ] 分析 max_alpha 参数的学习轨迹

**成功标准**: D 配置 Subject Consistency 显著优于 B 和 C

---

#### 阶段 3：系统 Benchmark + 论文撰写（Week 8-10）

**任务**:
- [ ] 在 2+ 环境（Atari 游戏 / Minecraft）上验证泛化性
- [ ] 完成 Anti-Drift 方法 head-to-head 对比表格
- [ ] 长序列测试：Drift 曲线对比（100/500/1000/2000 步）
- [ ] 人工评估（可选）：A/B 测试 5+ 参与者
- [ ] 错误案例分析（Failure Cases）
- [ ] 论文撰写

---

### 7.6 评测指标体系

**主要指标**（越高越好）:
| 指标 | 工具 | 意义 |
|---|---|---|
| Subject Consistency | VBench / DINO cosine sim | 主体一致性（核心）|
| Background Consistency | VBench / CLIP cosine sim | 背景一致性 |
| Motion Smoothness | VBench / AMFlow | 运动平滑度 |
| Dynamic Degree | VBench / RAFT | 防静态视频 |
| Aesthetic Quality | LAION Aesthetic Predictor | 视觉美感 |

**辅助指标**:
- LPIPS（感知相似度）：与参考视频对比
- FID/FVD：整体分布质量
- 用户研究（可选）：人工评估长期一致性

---

### 7.7 技术实现细节

#### 7.7.1 DINOv2 集成

```python
from transformers import AutoImageProcessor, AutoModel
import torch
import torch.nn.functional as F

class DINORewardModel:
    def __init__(self, model_name="facebook/dinov2-base", device="cuda"):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def get_features(self, images):
        """
        images: PIL Image list or tensor [T, C, H, W]
        returns: [T, D] normalized features
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # CLS token features
        features = outputs.last_hidden_state[:, 0, :]  # [T, D]
        return F.normalize(features, dim=-1)

    def compute_reward(self, frames):
        """
        frames: List of PIL Images (one chunk, e.g., 16 frames)
        returns: scalar reward in [0, 1]
        """
        features = self.get_features(frames)
        # Consecutive frame similarity
        sim = (features[:-1] * features[1:]).sum(dim=-1)  # [T-1]
        # Map from [-1,1] to [0,1]
        return ((sim + 1) / 2).mean().item()
```

#### 7.7.2 建议的 RL 训练框架选型

| 框架 | 优点 | 适用场景 |
|---|---|---|
| **Re-DMD（推荐）** | 无需通过奖励模型反向传播，奖励仅作加权因子，稳定 | Wan2.1 / DiffSynth 微调 |
| **DiffusionNFT** | 专为 Flow Matching 设计，WorldCompass 已验证 | HY-World 1.5 兼容环境 |
| **GRPO** | 简单易实现，但需要注意 KL 正则项的强度 | 快速原型验证 |

**Re-DMD 核心思路**:
```python
# 对于每个 rollout 样本 x，其奖励为 r
# Re-DMD 损失：用奖励加权分布匹配损失
loss_redmd = reward_weighted_dmd_loss(
    x_gen=generated_chunk,
    r=dino_reward,
    teacher=pretrained_model,  # Teacher 提供分布目标
    clip_range=0.2
)
```

---

### 7.8 预期结果与贡献

**预期改进**:
- VBench Subject Consistency: 基线 → 基线 + 2.0~3.5（参考 BAgger 的 +3.35）
- LPIPS@1000 步: 可学习噪声 < 固定高斯 < 无噪声
- Dynamic Degree: 不应显著下降（验证多奖励设计的有效性）
- 长序列（>500帧）的 Drift 曲线明显趋于平稳

**论文贡献**:
1. **Learned Adaptive Noise Augmentation**: 首次将可学习空间自适应噪声迁移到交互式世界模型
2. **DINO-Guided RL Anti-Drift**: 首次用语义级 DINO 奖励闭环优化世界模型的 anti-drift 能力
3. **Anti-Drift Benchmark**: 首个在统一基准上系统对比 5+ anti-drift 方法的工作
4. 实验验证"语义级监督 > 像素级监督"和"自适应噪声 > 固定噪声"两个关键假设

---

### 7.9 风险评估与 Plan B

| 风险 | 概率 | 影响 | 缓解 Plan B |
|---|---|---|---|
| DINO Reward Hacking（生成静态视频）| 高 | 高 | 立即加入 Dynamic Degree / RAFT 光流惩罚项 |
| 可学习噪声策略 RL 训练不稳定 | 中 | 高 | 先固定噪声策略做 DINO RL，确认 RL loop 稳定后再解锁 π_θ |
| RL 训练不稳定（奖励崩溃）| 中 | 高 | 降低 LR 至 1e-6，增大 EMA decay |
| 计算资源不足 | 中 | 中 | 优先用 DIAMOND（Atari，单卡可跑），大模型实验申请集群 |
| 可学习噪声 vs 固定高斯改进不显著 | 中 | 高 | Benchmark 贡献本身仍有价值，可调整论文重心到 DINO RL |
| DINO 对几何 Drift 不敏感 | 中 | 中 | 补充 DepthAnything 几何一致性奖励 |
| VBench 改进不显著 | 低 | 高 | 引入更细粒度 Drift 曲线（DINO sim vs step）作为评测 |

---

## 8. 参考文献

### 核心论文

1. **HY-World 1.5 Tech Report** (Tencent, Dec 2025)
   https://3d-models.hunyuan.tencent.com/world/world1_5/HYWorld_1.5_Tech_Report.pdf

2. **WorldCompass: RL for Long-Horizon World Models** (Tencent, Feb 2026)
   arxiv: 2602.09022

3. **HunyuanWorld 1.0** (Tencent, Jul 2025)
   arxiv: 2507.21809

4. **VBench: Comprehensive Benchmark Suite for Video Generative Models** (CVPR 2024)
   https://vchitect.github.io/VBench-project/

### Anti-Drift 方法

5. **GameNGen: Diffusion Models Are Real-Time Game Engines** (2024)
   arxiv: 2408.14837

6. **BAgger: Backwards Aggregation for Mitigating Drift** (2024)
   arxiv: 2512.12080

7. **Stable Video Infinity: Infinite-Length Video Generation with Error Recycling** (ICLR 2026 Oral)
   arxiv: 2510.09212

8. **Rolling Forcing: Autoregressive Long Video Diffusion in Real Time** (ICLR 2026)
   arxiv: 2509.25161

9. **Self-Forcing** (NeurIPS 2025 Spotlight)
   https://github.com/guandeh17/Self-Forcing

10. **RLVR-World: Training World Models with Reinforcement Learning** (2025)
    arxiv: 2505.13934

11. **Reward Forcing: Efficient Streaming Video Generation** (2024)
    arxiv: 2512.04678

12. **Video Generation Models Are Good Latent Reward Models (PRFL)** (2025)
    arxiv: 2511.21541

### DINO 相关

13. **DINO: Self-Distillation with No Labels** (Caron et al., 2021)
    arxiv: 2104.14294

14. **DINOv2: Learning Robust Visual Features without Supervision** (Oquab et al., 2023)
    arxiv: 2304.07193

15. **DINO-WM: World Models on Pre-trained Visual Features** (ICML 2025)
    arxiv: 2411.04983

16. **Back to the Features: DINO as Foundation for Video World Models** (Jul 2025)
    arxiv: 2507.19468

17. **Rewarding DINO: Predicting Dense Rewards with Vision Foundation Models** (Mar 2026)
    arxiv: 2603.16978

18. **DIVE: Taming DINO for Subject-Driven Video Editing** (2024)
    arxiv: 2412.03347

### 交互式世界模型

19. **DIAMOND: Diffusion for World Modeling** (NeurIPS 2024 Spotlight)
    arxiv: 2405.12399

20. **Oasis: A Universe in a Transformer** (Decart, 2024)
    https://oasis-model.github.io/

21. **GameGen-X: Interactive Open-world Game Video Generation** (ICLR 2025)
    arxiv: 2411.00769

22. **Genie: Generative Interactive Environments** (DeepMind, 2024)
    arxiv: 2402.15391

23. **Hunyuan-GameCraft** (Tencent, 2025)
    arxiv: 2506.17201

24. **Matrix-Game 2.0** (Skywork AI, 2025)
    arxiv: 2508.13009

25. **Vid2World** (2025)
    arxiv: 2505.14357

### Learned Noise 相关

26. **MuLAN: Diffusion Models With Learned Adaptive Noise** (NeurIPS 2024 Spotlight)
    https://openreview.net/forum?id=loMa99A4p8

27. **DDPO: Training Diffusion Models with Reinforcement Learning** (2023)
    arxiv: 2305.13301

### 综述与背景

28. **A Survey: Spatiotemporal Consistency in Video Generation** (2025)
    arxiv: 2502.17863

29. **Improving Video Generation with Human Feedback** (2025)
    arxiv: 2501.13918

---

*本 Proposal 生成于 2026-03-23，更新于 2026-03-23（加入 Noise Augmentation 调研与可学习噪声研究方向）。*
