# 基于 Wan2.2-TI2V-5B 的 LoRA 微调实现视频目标替换：可行性研究报告

> **日期**: 2026-03-22
> **研究方向**: 基于 LoRA 微调的蒙版引导视频目标替换
> **基础模型**: Wan2.2-TI2V-5B (5B 参数, Dense DiT)

---

## 目录

1. [研究概述与核心思路](#1-研究概述与核心思路)
2. [可行性判断](#2-可行性判断)
3. [Wan2.2-TI2V-5B 架构分析](#3-wan22-ti2v-5b-架构分析)
4. [相关工作对比](#4-相关工作对比)
5. [技术方案设计](#5-技术方案设计)
6. [训练策略](#6-训练策略)
7. [数据集构建](#7-数据集构建)
8. [评估方案](#8-评估方案)
9. [风险与缓解措施](#9-风险与缓解措施)
10. [研究创新点总结](#10-研究创新点总结)
11. [参考文献](#11-参考文献)

---

## 1. 研究概述与核心思路

### 1.1 目标

在 Wan2.2-TI2V-5B 模型上，通过 LoRA 微调输出层和中间层，使模型学会：

1. 接受**源视频首帧 + 蒙版**（标记待替换目标区域）
2. 接受**参考图**（替换目标的外观参考）
3. 利用**文本提示**进行语义引导和约束
4. 根据源视频的运动模式，生成目标被替换后的新视频

### 1.2 核心流程

```
输入:
  ├── 源视频首帧 (source_frame)
  ├── 目标蒙版 (mask) → 标记首帧中待替换区域
  ├── 参考图 (reference_image) → 替换物体的外观
  └── 文本提示 (prompt) → 语义引导

处理:
  1. VAE 编码: source_frame → z_source (48, 1, H/8, W/8)
  2. VAE 编码: reference_image → z_ref (48, 1, H/8, W/8)
  3. 蒙版处理: mask → m_latent (1, 1, H/8, W/8)
  4. 首帧融合: z_fused = z_source * (1-m) + noise * m  (蒙版区域用噪声)
  5. 参考图注入: 沿时间维度拼接 z_ref 或通过 cross-attention 注入
  6. DiT 去噪: 30层 Transformer 生成视频潜码
  7. VAE 解码: 输出替换后的视频

输出:
  └── 替换后视频 (replaced_video)
```

---

## 2. 可行性判断

### 结论：**可行，且有明确创新空间**

| 维度 | 评估 | 依据 |
|------|------|------|
| **技术可行性** | ✅ 高 | LoRA-Edit 已证明蒙版感知 LoRA 微调在 Wan2.1-I2V 上有效；DreamSwapV 在 Wan2.1 DiT 上实现了目标替换 |
| **架构兼容性** | ✅ 高 | TI2V-5B 的 VAE 融合机制天然支持首帧蒙版操作；48通道潜码空间提供足够的信息容量 |
| **硬件可行性** | ✅ 高 | 5B 模型 LoRA 微调约需 20-24GB 显存，单张 RTX 5090 (32GB) 即可训练和推理 |
| **数据可行性** | ✅ 中高 | 可通过 SAM2 + 视频数据集自监督构建训练对；DAVIS 可用作评估集 |
| **创新空间** | ✅ 高 | 现有工作要么全参数微调(DreamSwapV)，要么逐视频优化(LoRA-Edit)，要么通用但效果弱(VACE)；基于 LoRA 的通用目标替换方案是明确空白 |

---

## 3. Wan2.2-TI2V-5B 架构分析

### 3.1 整体架构

```
Wan2.2-TI2V-5B
├── Text Encoder: umt5-xxl (4096-dim embeddings)
├── VAE: WanVideoVAE38 (z_dim=48, compression=4×16×16)
└── DiT: WanModel (30 blocks, dim=3072, 24 heads)
    ├── Patch Embedding: Conv3d(48→3072, kernel=(1,2,2))
    ├── Text Embedding: Linear(4096→3072) → GELU → Linear(3072→3072)
    ├── Time Embedding: Sinusoidal(256) → Linear(256→3072) → SiLU → Linear(3072→3072)
    ├── 30× DiTBlock:
    │   ├── Self-Attention (Q,K,V,O projections + 3D RoPE)
    │   ├── Cross-Attention (Q,K,V,O projections, text conditioning)
    │   ├── FFN (Linear(3072→14336) → GELU → Linear(14336→3072))
    │   └── AdaLN Modulation (6 params per block from timestep)
    └── Head: LayerNorm → Linear(3072→192) → Unpatchify
```

### 3.2 TI2V 图像融合机制（关键）

TI2V-5B **不使用** CLIP 图像编码器，而是采用 **VAE 潜码融合**：

```python
# WanVideoUnit_ImageEmbedderFused 的核心逻辑
z = vae.encode(input_image)          # (1, 48, 1, H/8, W/8)
latents[:, :, 0:1] = z              # 首帧噪声直接替换为图像潜码
# 去噪时首帧 timestep=0, 其余帧 timestep=t (separated_timestep=True)
```

**这个机制为蒙版操作提供了天然接口**：我们可以在潜码空间对首帧进行蒙版处理，让蒙版区域保持噪声，非蒙版区域保留图像信息。

### 3.3 关键配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `dim` | 3072 | DiT 隐藏维度 |
| `in_dim` / `out_dim` | 48 | VAE 潜码通道数 |
| `ffn_dim` | 14336 | FFN 隐藏维度 |
| `num_heads` | 24 | 注意力头数 (128-dim/head) |
| `num_layers` | 30 | Transformer 块数 |
| `patch_size` | (1, 2, 2) | 时间不分patch，空间2×2 |
| `text_dim` | 4096 | umt5-xxl 文本嵌入维度 |
| `fuse_vae_embedding_in_latents` | True | 图像通过 VAE 潜码融合 |
| `seperated_timestep` | True | 首帧与后续帧分离的时间步 |

### 3.4 可微调的 LoRA 目标层

| 层类型 | 参数路径 | 作用 | 推荐优先级 |
|--------|---------|------|-----------|
| **Self-Attention Q/K/V/O** | `blocks.{i}.self_attn.{q,k,v,o}` | 控制时空特征注意力模式 | ⭐⭐⭐ 最高 |
| **Cross-Attention Q/K/V/O** | `blocks.{i}.cross_attn.{q,k,v,o}` | 控制文本引导对齐 | ⭐⭐⭐ 最高 |
| **FFN** | `blocks.{i}.ffn.{0,2}` | 非线性特征变换 | ⭐⭐ 高 |
| **Head Output** | `head.head` | 最终噪声预测 | ⭐⭐ 高 |
| **Time Embedding** | `time_embedding` | 时间步编码 | ⭐ 中 |
| **Text Embedding** | `text_embedding` | 文本条件处理 | ⭐ 中 |
| **Modulation** | `blocks.{i}.modulation` | 时间步调制 | ⭐ 中 |

---

## 4. 相关工作对比

### 4.1 核心相关论文

| 论文 | 会议/年份 | 方法 | 与本方案的关键区别 |
|------|----------|------|-------------------|
| **VACE** | ICCV 2025 | Context Adapter + VCU (Text+Frame+Mask) | 通用多任务设计，目标替换效果偏弱（静态/错误主体）；使用完整 Adapter 而非 LoRA |
| **DreamSwapV** | arXiv 2025 | 全参数微调 Wan2.1 DiT + 条件融合模块 | 全参数微调（非 LoRA）；需两阶段训练；计算成本高 |
| **LoRA-Edit** | arXiv 2025 | 逐视频 LoRA 微调 + 蒙版感知 | 逐视频优化（每个视频训练100步）；无参考图支持；非通用模型 |
| **AnyV2V** | arXiv 2024 | 免训练，首帧编辑+I2V+特征注入 | 无训练，依赖外部图像编辑器；无法直接参考图引导替换 |
| **I2VEdit** | SIGGRAPH Asia 2024 | Motion LoRA + 注意力匹配 | 传播首帧编辑结果；不支持参考图引导 |
| **Phantom** | ICCV 2025 | 跨模态对齐三元组数据 | 主体驱动视频生成（非编辑/替换） |
| **AVID** | CVPR 2024 | 文本引导视频修复 | 仅文本引导，无参考图 |
| **EraserDiT** | arXiv 2025 | DiT视频修复+循环位置偏移 | 目标移除/背景填充，非替换 |

### 4.2 差异化定位

```
                    通用性 ↑
                      │
              VACE ●  │
                      │        ← 本方案目标位置
                      │           (LoRA + 蒙版 + 参考图)
                      │
    LoRA-Edit ●       │       ● DreamSwapV
    (逐视频)          │       (全参数微调)
                      │
                      └─────────────────────→ 参数效率 ↑
```

**本方案的独特定位**: 在 VACE（通用但弱）和 DreamSwapV（强但贵）之间，用 LoRA 实现参数高效的通用目标替换。

---

## 5. 技术方案设计

### 5.1 总体架构设计

```
┌─────────────────────────────────────────────────────┐
│                  训练阶段                            │
│                                                     │
│  Source Video ──→ SAM2 分割 ──→ (Frame₀, Mask, Ref) │
│                                                     │
│  Frame₀ + Mask ──→ VAE Encode ──→ z_masked          │
│  Reference     ──→ VAE Encode ──→ z_ref             │
│  Text Prompt   ──→ T5 Encode  ──→ c_text            │
│                                                     │
│  z_fused = Concat([z_ref, z_masked+noise], dim=T)   │
│     ↓                                               │
│  Frozen DiT + LoRA ──→ 噪声预测                      │
│     ↓                                               │
│  Flow Matching Loss (原始视频作为GT)                   │
│                                                     │
│  可训练参数: LoRA weights + Mask Projection Layer     │
└─────────────────────────────────────────────────────┘
```

### 5.2 输入条件设计（三种候选方案）

#### 方案 A: 时间维度拼接（推荐）

```python
# 参考图作为"第0帧"，蒙版首帧作为"第1帧"的起始条件
z_ref = vae.encode(reference_image)     # (1, 48, 1, H/8, W/8)
z_source = vae.encode(source_frame)     # (1, 48, 1, H/8, W/8)
mask_latent = downsample(mask)          # (1, 1, 1, H/8, W/8)

# 蒙版区域用噪声替代
z_masked = z_source * (1 - mask_latent) + noise * mask_latent

# 时间拼接: [参考图, 蒙版首帧, 后续帧...]
latents[:, :, 0:1] = z_ref             # 参考图作为"锚定帧"
latents[:, :, 1:2] = z_masked          # 蒙版首帧

# 自定义时间步:
# 参考图帧 → timestep=0 (保持不变)
# 蒙版首帧 → timestep=0 或低值 (部分保持)
# 后续帧   → timestep=t (正常去噪)
```

**优点**: 完全兼容现有架构，不需要修改 DiT 输入维度
**缺点**: 增加了序列长度，略增计算量

#### 方案 B: 通道维度拼接

```python
# 在潜码通道维度拼接蒙版信息
# 原始 in_dim=48, 扩展为 48+1=49 (加蒙版通道)
z_input = torch.cat([z_masked, mask_latent], dim=1)  # (1, 49, F, H/8, W/8)

# 需要修改 Patch Embedding 的输入通道: Conv3d(49→3072)
# 新增的1个通道权重随机初始化或零初始化
```

**优点**: 蒙版信息直接编码进每个 patch
**缺点**: 需要修改 Patch Embedding 层（少量额外参数，非 LoRA）

#### 方案 C: VACE 风格 Context Adapter

```python
# 添加轻量 Context Block (参考 VACE 架构)
# 14 个 Context Block 均匀分布在 30 层 DiT 中
# 输入: [参考图潜码, 蒙版首帧潜码, 蒙版序列]
# 输出: 加性信号注入 DiT 的对应层
```

**优点**: 不修改原始 DiT，最灵活
**缺点**: 引入较多额外参数（超出 LoRA 范畴），训练复杂度高

### 5.3 推荐方案: A + B 组合

```
方案 A（时间维度拼接参考图）+ 方案 B（通道维度拼接蒙版）
```

具体实现：

1. **蒙版注入**: 扩展 Patch Embedding 输入通道 48→49，第49通道为蒙版（需额外训练此层）
2. **参考图注入**: 沿时间维度在首帧前拼接参考图潜码，该帧 timestep=0
3. **LoRA 微调**: 在 Self-Attention、Cross-Attention、FFN 和 Head 层注入 LoRA

### 5.4 LoRA 配置建议

```python
lora_config = {
    "rank": 64,                    # 较高 rank 以学习蒙版+参考图的复杂关系
    "alpha": 64,                   # alpha = rank (标准设置)
    "target_modules": [
        # Self-Attention (30 blocks × 4 projections)
        "blocks.*.self_attn.q",
        "blocks.*.self_attn.k",
        "blocks.*.self_attn.v",
        "blocks.*.self_attn.o",
        # Cross-Attention (30 blocks × 4 projections)
        "blocks.*.cross_attn.q",
        "blocks.*.cross_attn.k",
        "blocks.*.cross_attn.v",
        "blocks.*.cross_attn.o",
        # FFN (30 blocks × 2 linear)
        "blocks.*.ffn.0",
        "blocks.*.ffn.2",
        # Head output
        "head.head",
    ],
    "dropout": 0.05,
}

# 估算可训练参数量:
# Self-Attn: 30 × 4 × 2 × 3072 × 64 ≈ 47M
# Cross-Attn: 30 × 4 × 2 × 3072 × 64 ≈ 47M
# FFN: 30 × 2 × 2 × (3072+14336) × 64 ≈ 133M
# Head: 2 × 3072 × 64 ≈ 0.4M
# 总计: ~227M 可训练参数 (约占 5B 的 4.5%)
```

### 5.5 损失函数设计

```python
def compute_loss(model, batch):
    """
    蒙版感知 Flow Matching 损失
    """
    # 准备输入
    z_gt = vae.encode(target_video)              # GT 视频潜码
    z_ref = vae.encode(reference_image)           # 参考图潜码
    z_source = vae.encode(source_first_frame)     # 源视频首帧潜码
    mask = preprocess_mask(object_mask)            # 目标蒙版
    text_emb = t5.encode(prompt)                  # 文本嵌入

    # Flow Matching 采样
    t = torch.rand(batch_size)                    # 随机时间步
    noise = torch.randn_like(z_gt)
    z_t = (1 - t) * z_gt + t * noise             # 插值

    # 模型预测
    v_pred = model(z_t, t, text_emb, z_ref, z_source, mask)
    v_gt = noise - z_gt                           # GT 速度场

    # 蒙版加权损失: 蒙版区域（替换目标）权重更高
    weight_map = torch.ones_like(mask)
    weight_map[:, :, 0:1] = 1 + mask * 2.0       # 首帧蒙版区域 3x 权重

    loss = F.mse_loss(v_pred * weight_map, v_gt * weight_map)

    # 可选: 参考图保真度损失
    # loss_ref = reference_fidelity_loss(decoded_masked_region, reference_image)

    return loss
```

---

## 6. 训练策略

### 6.1 两阶段训练

#### 阶段一: 蒙版感知预训练 (Mask-Aware Pretraining)

- **目标**: 让模型学会根据蒙版区分"保留"与"生成"区域
- **数据**: 自监督构建 — 从视频中分割目标，蒙版该目标，用同一目标的裁剪图作为参考
- **训练设置**:
  - 学习率: 1e-4
  - Batch Size: 1-2 (梯度累积 4-8 步)
  - 训练步数: 10,000-20,000
  - 分辨率: 480×832 或 384×672 (降低以节省显存)
  - 帧数: 41-81 帧 (2-5 秒视频)
  - LoRA Rank: 64
  - 优化器: AdamW, weight_decay=0.01

#### 阶段二: 目标替换微调 (Object Replacement Fine-Tuning)

- **目标**: 让模型学会用不同外观的参考图替换蒙版区域目标
- **数据**: 跨视频目标替换对 — 将 A 视频的目标用 B 视频的目标替换
- **训练设置**:
  - 学习率: 5e-5 (降低)
  - 训练步数: 5,000-10,000
  - 其他同阶段一
  - 增加参考图增强 (随机缩放/旋转/颜色偏移)

### 6.2 显存优化

```
预估显存需求 (单张 RTX 5090, 32GB):
├── DiT 模型 (BF16): ~10 GB
├── VAE (BF16): ~2 GB
├── T5 Text Encoder: ~4 GB (可 offload 到 CPU)
├── LoRA 参数 + 优化器状态: ~2 GB
├── 激活值 (gradient checkpointing): ~8 GB
└── 总计: ~26 GB ✅ 可行

优化策略:
├── Gradient Checkpointing: 必须启用
├── Mixed Precision: BF16 训练
├── T5 CPU Offload: 编码文本后卸载
├── Tiled VAE: 分块编解码大分辨率视频
└── 梯度累积: 小 batch + 多步累积
```

---

## 7. 数据集构建

### 7.1 自监督训练数据构建流程

```
Step 1: 视频收集
  ├── WebVid-10M (大规模预训练)
  ├── Panda-70M (已在服务器上: ~/Panda-70M)
  └── YouTube-VOS / SA-V (带标注)

Step 2: 目标分割 (SAM2)
  ├── 输入: 视频序列
  ├── 输出: 每帧的目标蒙版
  └── 自动/半自动标注

Step 3: 训练对构建
  ├── 参考图: 首帧目标裁剪 + 随机增强
  ├── 蒙版首帧: 首帧 + 目标区域蒙版
  ├── 目标视频: 原始视频 (GT)
  └── 文本: 视频描述 (BLIP-2/InternVL 自动生成)

Step 4: 跨视频替换对 (阶段二)
  ├── 同类别目标匹配 (如 狗↔狗, 车↔车)
  ├── 对齐尺度和位置
  └── 生成替换 GT (可用 VACE/DreamSwapV 生成伪 GT)
```

### 7.2 可用数据集

| 数据集 | 位置 | 用途 |
|--------|------|------|
| **DAVIS** | `~/../datasets/DAVIS/` | 评估集；480p 视频 + 目标分割标注 |
| **Panda-70M** | `~/Panda-70M/` | 大规模视频预训练数据 |
| **YouTube-VOS** | 需下载 | 大规模视频目标分割，带蒙版标注 |
| **SA-V (SAM 2)** | 需下载 | 视频分割标注 |

### 7.3 参考图增强策略

防止模型学到"复制粘贴"捷径：

```python
reference_augmentation = {
    "random_resize": (0.8, 1.2),         # 随机缩放
    "random_rotation": (-15, 15),        # 随机旋转
    "color_jitter": {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1,
    },
    "random_horizontal_flip": 0.5,       # 随机水平翻转
    "random_crop_pad": 0.1,              # 随机裁剪/填充
}
```

---

## 8. 评估方案

### 8.1 定量指标

| 指标 | 评估维度 | 计算方法 |
|------|---------|---------|
| **DINO Similarity** | 参考图保真度 | 参考图与生成目标区域的 DINO 特征余弦相似度 |
| **CLIP Score** | 文本-视频对齐 | 文本与视频帧的 CLIP 相似度 |
| **Background PSNR/SSIM** | 背景保持 | 源视频与生成视频非蒙版区域的 PSNR/SSIM |
| **FVD** | 整体视频质量 | Fréchet Video Distance |
| **Subject Consistency** | 目标跨帧一致性 | VBench 子指标 |
| **Motion Smoothness** | 运动平滑度 | VBench 子指标 |
| **FVMD** | 运动一致性 | 关键点追踪的速度/加速度场比较 |

### 8.2 定性评估

- **用户研究**: 参考图保真度、运动自然度、背景保持、整体视觉质量
- **消融实验**: LoRA rank、目标层选择、蒙版策略、损失权重

### 8.3 评估数据集

1. **DAVIS** (已有): 50 个视频序列，480p，带目标分割标注
2. **自建 Benchmark**: 参考 DreamSwapV-Benchmark，构建 100 个视频、多类别目标替换测试集

### 8.4 对比基线

| 基线 | 类型 |
|------|------|
| VACE (Wan2.1-VACE-14B) | 通用视频编辑 |
| AnyV2V | 免训练首帧编辑传播 |
| LoRA-Edit | 逐视频 LoRA 优化 |
| DreamSwapV (如有代码) | 全参数微调目标替换 |

---

## 9. 风险与缓解措施

### 9.1 主要风险

| 风险 | 严重度 | 概率 | 缓解措施 |
|------|--------|------|---------|
| **LoRA 表达能力不足** | 高 | 中 | 提升 rank 至 128；对 Patch Embedding 和蒙版投影层进行全参数训练（仅增加少量参数） |
| **参考图身份丢失** | 高 | 中高 | 增加参考图保真度损失；尝试 self-attention isolation（参考图仅提供 KV） |
| **运动不连贯** | 中 | 中 | 引入光流/深度图作为额外条件；使用运动平滑度正则化 |
| **背景泄露/污染** | 中 | 中低 | 蒙版加权损失确保非蒙版区域严格保持 |
| **训练数据质量** | 中 | 中 | 数据清洗 + 多轮筛选；使用 VLM 评估生成的训练对质量 |
| **跨域泛化差** | 中 | 中 | 使用多样化训练数据；增加文本条件强度 |

### 9.2 备选技术路径

如果纯 LoRA 方案效果不佳，可逐步升级：

```
Plan A: 纯 LoRA (rank=64, ~227M params)
  ↓ 效果不足
Plan B: LoRA + Mask Projection Layer (~230M params)
  ↓ 效果不足
Plan C: LoRA + Lightweight Reference Adapter (~300M params)
  ↓ 效果不足
Plan D: LoRA + Context Adapter (类 VACE, ~500M params)
```

---

## 10. 研究创新点总结

### 10.1 核心贡献

1. **参数高效的视频目标替换**: 首次在 TI2V 架构上使用 LoRA 实现通用蒙版引导目标替换，可训练参数仅占 4.5%
2. **TI2V VAE 融合机制的蒙版扩展**: 利用 TI2V-5B 独特的 VAE 潜码融合机制，自然集成蒙版操作，无需重新设计条件注入
3. **两阶段高效训练策略**: 自监督蒙版感知预训练 → 跨目标替换微调，降低数据需求
4. **蒙版感知 Flow Matching 损失**: 区域加权损失函数平衡替换区域生成质量与背景保持

### 10.2 与现有工作的差异化

| 维度 | VACE | DreamSwapV | LoRA-Edit | **本方案** |
|------|------|-----------|-----------|-----------|
| 参数效率 | 低 (Context Adapter) | 低 (全参数) | 高 (LoRA) | **高 (LoRA)** |
| 通用性 | 高 (多任务) | 中 (目标替换) | 低 (逐视频) | **中高 (目标替换)** |
| 参考图支持 | ✅ | ✅ | ❌ | **✅** |
| 蒙版引导 | ✅ | ✅ | ✅ | **✅** |
| 基础模型 | Wan2.1 | Wan2.1 | Wan2.1 | **Wan2.2 TI2V-5B** |
| 推理成本 | 高 (14B+Adapter) | 高 (14B 全参) | 中 (I2V+LoRA) | **低 (5B+LoRA)** |
| 训练成本 | 极高 (多任务) | 高 (两阶段全参) | 极低 (逐视频) | **中低 (两阶段LoRA)** |

### 10.3 潜在论文标题方向

- "MaskLoRA: Parameter-Efficient Video Object Replacement via Mask-Aware LoRA Fine-Tuning on TI2V Models"
- "LoRA-Swap: Efficient Reference-Guided Video Object Replacement with Masked Latent Fusion"
- "Towards Efficient Video Object Replacement: LoRA Adaptation of Text-Image-to-Video Diffusion Models"

---

## 11. 参考文献

### 核心参考

1. **VACE**: Jiang et al., "VACE: All-in-One Video Creation and Editing," ICCV 2025. [arXiv:2503.07598](https://arxiv.org/abs/2503.07598)
2. **DreamSwapV**: "DreamSwapV: Mask-guided Subject Swapping for Any Customized Video Editing," 2025. [arXiv:2508.14465](https://arxiv.org/abs/2508.14465)
3. **LoRA-Edit**: "LoRA-Edit: Controllable First-Frame-Guided Video Editing via Mask-Aware LoRA Fine-Tuning," 2025. [arXiv:2506.10082](https://arxiv.org/abs/2506.10082)
4. **Wan Technical Report**: [arXiv:2503.20314](https://arxiv.org/abs/2503.20314)

### 视频编辑与修复

5. **AnyV2V**: "A Tuning-Free Framework For Any Video-to-Video Editing Tasks," 2024. [arXiv:2403.14468](https://arxiv.org/abs/2403.14468)
6. **I2VEdit**: "First-Frame-Guided Video Editing via Image-to-Video Diffusion Models," SIGGRAPH Asia 2024. [arXiv:2405.16537](https://arxiv.org/abs/2405.16537)
7. **AVID**: "Any-Length Video Inpainting with Diffusion Model," CVPR 2024. [arXiv:2312.03816](https://arxiv.org/abs/2312.03816)
8. **EraserDiT**: "Fast Video Inpainting with Diffusion Transformer Model," 2025. [arXiv:2506.12853](https://arxiv.org/abs/2506.12853)

### 主体一致性与参考引导

9. **Phantom**: "Subject-Consistent Video Generation via Cross-Modal Alignment," ICCV 2025. [arXiv:2502.11079](https://arxiv.org/abs/2502.11079)
10. **SUGAR**: "Zero-Shot Subject-Driven Video Customization," 2024.

### LoRA 与高效微调

11. **LoRA Fine-Tuning for Wan2.1 I2V**: [arXiv:2510.27364](https://arxiv.org/abs/2510.27364)
12. **LoRA Recycle**: "Unlocking Tuning-Free Few-Shot Adaptability," CVPR 2025.

### 评估

13. **VBench**: "Comprehensive Benchmark Suite for Video Generative Models," CVPR 2024 Highlight.
14. **VBench-2.0**: [arXiv:2503.21755](https://arxiv.org/abs/2503.21755)
15. **FVMD**: Motion consistency metrics via keypoint tracking.

### 架构与加速

16. **FramePack**: [arXiv:2504.12626](https://arxiv.org/abs/2504.12626), NeurIPS 2025 Spotlight.
17. **TeaCache**: "Timestep Embedding Tells Its Time to Cache," CVPR 2025.

---

## 附录 A: 关键代码文件路径

| 文件 | 路径 | 说明 |
|------|------|------|
| DiT 模型 | `DiffSynth-Studio/diffsynth/models/wan_video_dit.py` | DiT 架构实现 |
| VAE | `DiffSynth-Studio/diffsynth/models/wan_video_vae.py` | VAE38 编解码器 |
| Pipeline | `DiffSynth-Studio/diffsynth/pipelines/wan_video.py` | 推理管线 |
| 模型配置 | `DiffSynth-Studio/diffsynth/configs/model_configs.py` | TI2V-5B 参数配置 |
| 训练脚本 | `DiffSynth-Studio/examples/wanvideo/model_training/train.py` | 训练入口 |
| 推理示例 | `DiffSynth-Studio/examples/wanvideo/model_inference/Wan2.2-TI2V-5B.py` | 推理示例 |
| VACE 实现 | `DiffSynth-Studio/diffsynth/models/wan_video_vace.py` | VACE 参考实现 |
| TI2V-5B 模型 | `~/Wan2.2/Wan2.2-TI2V-5B/` (远程服务器) | 预训练权重 |
| DAVIS 数据集 | `~/../datasets/DAVIS/` (远程服务器) | 评估数据集 |

## 附录 B: 实施时间线建议

| 阶段 | 任务 | 预期产出 |
|------|------|---------|
| **阶段 1** | 环境搭建 + 基线推理 | TI2V-5B 正常推理验证 |
| **阶段 2** | 蒙版注入机制实现 | 修改 Pipeline 支持蒙版+参考图输入 |
| **阶段 3** | LoRA 训练框架搭建 | 自监督数据构建 + 训练循环 |
| **阶段 4** | 阶段一训练 | 蒙版感知预训练完成 |
| **阶段 5** | 阶段二训练 + 消融实验 | 目标替换微调 + 超参搜索 |
| **阶段 6** | 评估 + 对比实验 | DAVIS 评估 + 基线对比 |
| **阶段 7** | 论文撰写 | 完整论文 |
