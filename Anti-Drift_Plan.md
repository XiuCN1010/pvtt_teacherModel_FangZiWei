# 实验可行性分析：DINO Anti-Drift 方向

**日期**: 2026-03-28
**基于**: `016_drift_quantification.md` + `Proposal_DINO_AntiDrift_WorldModel.md`
**目的**: 梳理验证 Proposal 可行性与性能优化所需的数据、实验角度与具体方案

---

## 一、需要获取的数据

### 1.1 训练数据（世界模型微调用）

所需数据取决于选择的基础世界模型：

| 基础模型 | 所需数据 | 获取方式 | 备注 |
|---|---|---|---|
| **DIAMOND (推荐起步)** | Atari 游戏录像 (state-action pairs) | 开源，用 RL Agent 自动采集 | **单卡可跑，零成本启动** |
| **Oasis 500M** | Minecraft 游戏录像 | 开源数据集，或用 MineRL 环境自动采集 | 中等资源需求 |
| **Wan2.1 + DiffSynth** | 通用视频 + 动作标注 | 本地已有代码 (`/e/IP-2026-Spring_Source/`) | 大模型，需集群资源 |

**建议先用 DIAMOND**，数据获取成本最低、单卡可跑，适合验证可行性。

### 1.2 评测数据（Drift 量化用）

016 报告已建立 Drift 量化方法（3 个场景、61 帧），但需要扩展：

- **更长的自回归序列**：016 只测了 61 帧（4 chunk），Proposal 要求测 **100 / 500 / 1000 / 2000 步**的 Drift 曲线
- **更多场景类型**：016 只有 3 个场景（Beach / Mountain / Garden），需要覆盖不同复杂度的场景（简单静态 → 复杂动态），建议至少 **8~10 个场景**
- **Ground truth 参考视频**（可选）：如果要算 LPIPS / FID / FVD，需要真实游戏录像作为参考分布

### 1.3 预训练模型权重

| 模型 | 用途 | 来源 |
|---|---|---|
| **DINOv2 ViT-B/14** | Anti-Drift 核心奖励信号 | `facebook/dinov2-base`，HuggingFace 直接下载 |
| **基础世界模型** | 微调基座 | DIAMOND / Oasis 开源 checkpoint |
| **RAFT** | 光流估计，防静态视频 (Dynamic Degree) | 开源 |
| **HPSv3**（可选） | 视觉质量奖励 | 开源 |
| **DepthAnythingV3**（可选） | 几何一致性奖励 | 开源 |

---

## 二、验证可行性的实验

核心问题：**DINO Reward 能否有效驱动 RL 优化 anti-drift？**

建议分三个层级递进验证：

### 2.1 角度 1：DINO 作为 Drift 检测器的灵敏度验证（零训练成本）

016 报告已部分完成（3 个场景、61 帧、DINO sim 单调下降 2.5%~11.7%）。需要进一步确认：

**实验 1a — 更长序列的 DINO 衰减曲线**

- 生成 8 / 16 chunk 视频（约 128 / 256 帧）
- 绘制 DINO sim vs frame index 曲线
- **验证目标**：DINO sim 是否持续单调下降？还是在某个点饱和？
- 如果持续下降 → 说明 DINO 在长序列上仍是有效信号
- 如果饱和 → 可能需要补充其他指标

**实验 1b — 场景区分度**

- 覆盖更多场景（简单 → 复杂），计算各场景的 DINO 下降速率
- 请人工标注 drift 严重程度排名
- 计算 DINO 下降速率与人类感知排名的 **Spearman 相关系数**
- **验证目标**：相关系数 > 0.7 说明 DINO 确实捕捉到了有意义的退化

**实验 1c — 与其他指标的相关性**

- 同时计算 LPIPS、PSNR、CLIP sim 的衰减曲线
- 计算 DINO sim 下降与 LPIPS 上升之间的 **Pearson 相关系数**
- **验证目标**：确认 DINO 不是孤立信号，而是与感知质量退化正相关

> 这些实验**零训练成本**，只需要生成视频 + 跑前向推理。

### 2.2 角度 2：DINO Reward 的可优化性验证（轻量 RL）

这是可行性的**关键门槛**：

**实验 2a — 最简 RL Loop**

- 在 DIAMOND 上搭建 RL 训练环境
- 使用固定高斯噪声（GameNGen 方式, α ~ U[0, 0.7]）
- 以 DINO consistency 作为 reward
- 跑少量 epoch（~50 epoch），观察 reward 曲线

**验证目标**：
- DINO reward 随训练**单调上升** → 说明 RL 可以有效优化该信号
- 同时监控生成视频的 Dynamic Degree（RAFT 光流幅度）
  - 如果 Dynamic Degree 显著下降 → **Reward Hacking 发生**，需要加 motion 约束
  - 如果 Dynamic Degree 稳定 → 无 hacking 风险

**实验 2b — DINO Reward vs MSE Loss 对比**

- 同样的 noise augmentation 设置
- 分别用 DINO reward 和 MSE loss 训练
- 对比两者在 VBench Subject Consistency 上的表现

**成功标准**（来自 Proposal 阶段 1）：DINO reward 版本的 Subject Consistency 优于 MSE 版本 **≥ 1.0**

> 这一阶段回答："DINO 语义奖励是否比像素级损失更适合做 anti-drift 监督？"

### 2.3 角度 3：Reward Hacking 风险的早期检测

016 报告在 4.3 节已做风险评估，需要实验验证：

**实验 3a — Reward Hacking 检测**

- 在 RL 训练过程中，每 N epoch 生成一批评测视频
- 同时记录：DINO reward / Dynamic Degree / Aesthetic Quality
- 绘制三者随训练步数的变化曲线
- **警戒线**：Dynamic Degree 下降超过 10% 即触发 motion 约束

**实验 3b — 多奖励权重扫描**

- 扫描 λ₁（DINO 权重）和 λ₂（motion 权重）的组合
- 建议网格：λ₁ ∈ {0.3, 0.5, 0.7, 0.9}，λ₂ ∈ {0.1, 0.2, 0.3, 0.5}
- 对每个组合，记录 Subject Consistency 和 Dynamic Degree
- 绘制 **Pareto 前沿**，找到最优权重配比

---

## 三、验证性能优化的实验

一旦可行性确认，以下实验验证"我们的方法比现有方案更好"：

### 3.1 实验 A：核心消融矩阵（Proposal 阶段 2 核心）

| 配置 | Noise 类型 | 奖励/损失 | 预期排名 |
|---|---|---|---|
| A: No augmentation | 无 | MSE | 最差（Drift 严重） |
| B: GameNGen 复现 | 固定高斯 α~U[0,0.7] | MSE | 中等改善 |
| C: 固定高斯 + DINO RL | 固定高斯 | DINO Reward | 较好改善 |
| **D: Learned Noise + DINO RL (Ours)** | **可学习自适应 π_θ** | **DINO Reward** | **最佳** |
| E: Diffusion Forcing | Per-frame noise | MSE | 对比方法 |
| F: Self-Forcing | Rollout training | MSE | 对比方法 |

**关键对比关系**：
- **C > B** → 验证假设 1：语义级监督 (DINO) 优于像素级监督 (MSE)
- **D > C** → 验证假设 2：自适应噪声优于固定噪声
- **D > E, F** → 验证整体方案优于其他 anti-drift 路线

**评测指标**（对每个配置都要测）：

| 指标 | 工具 | 越高越好？ |
|---|---|---|
| Subject Consistency | VBench / DINO cosine sim | 是 |
| Background Consistency | VBench / CLIP cosine sim | 是 |
| Motion Smoothness | VBench / AMFlow | 是 |
| Dynamic Degree | VBench / RAFT | 是（防静态） |
| Aesthetic Quality | LAION Aesthetic Predictor | 是 |
| LPIPS@{100,500,1K} 步 | 与参考帧的感知距离 | 否（越低越好） |

### 3.2 实验 B：长序列 Drift 曲线对比

- 对消融矩阵中的 A~F 所有配置，分别生成 **2000 步**的长序列
- 在 100 / 500 / 1000 / 2000 步处分别绘制 DINO sim 衰减曲线
- 016 报告的曲线绘制方法可以直接复用

**关键看点**：
- 我们的方法 (D) 是否让衰减曲线显著**趋于平稳**（slope 接近 0）？
- 在哪个时间点上，各方法开始出现明显差异？
- 预期：A 在 ~100 步开始明显退化，B/C 在 ~500 步，D 在 >1000 步仍保持稳定

### 3.3 实验 C：噪声策略网络的可解释性分析

- **可视化学到的 noise map**：
  - 对不同类型的游戏帧（天空、地面、UI、移动物体），可视化 π_θ 输出的空间噪声强度图
  - **预期**：高频区域（移动物体边缘）噪声强 → 更强纠错；低频区域（天空）噪声弱 → 保持稳定
- **max_alpha 参数的学习轨迹**：
  - 从 0.7 初始化后如何变化？收敛到什么值？
- **Noise map 的时序变化**：
  - 随着自回归步数增加，噪声策略是否自适应地增强噪声？

> 这些分析不直接验证性能，但对论文的 insight 和说服力至关重要。

### 3.4 实验 D：跨环境泛化验证（Proposal 阶段 3）

| 环境 | 模型 | 目的 |
|---|---|---|
| Atari (DIAMOND) | DIAMOND | 主实验环境 |
| Minecraft (Oasis) | Oasis 500M | 泛化验证 1 |
| 通用视频 (Wan2.1) | Wan2.1 + DiffSynth | 泛化验证 2（大模型） |

**验证目标**：方法不是特定于某个游戏/环境的，在不同视觉复杂度的环境中均有效。

---

## 四、推荐的实验路线图

```
Step 0（零成本，~1 周）
├── 扩展 016 的 Drift 量化到更长序列（8/16 chunk）
├── 更多场景的 DINO 衰减曲线
├── DINO vs LPIPS/PSNR 相关性分析
└── 产出：确认 DINO 在长序列上仍是有效信号

Step 1（低成本，~2 周）
├── DIAMOND 环境搭建 + GameNGen baseline 复现
├── 最简 RL loop + DINO reward 训练
├── DINO reward vs MSE loss 对比
├── Reward Hacking 早期检测
└── 产出：可行性确认报告

Step 2（核心实验，~3 周）
├── 实现 AdaptiveNoisePolicy 网络
├── 消融矩阵 A~F 全量实验
├── 长序列 Drift 曲线对比
├── Noise map 可解释性分析
├── 多奖励权重扫描 + Pareto 前沿
└── 产出：核心实验结果，验证性能优化

Step 3（泛化 + 论文，~3 周）
├── 跨环境验证（Oasis / Wan2.1）
├── Anti-Drift Benchmark 整理
├── 人工评估（可选，A/B 测试 5+ 参与者）
├── 失败案例分析
└── 产出：论文初稿
```

---

## 五、最急需的资源清单

| 优先级 | 资源 | 用途 | 获取难度 |
|---|---|---|---|
| **P0** | DIAMOND 开源环境 + checkpoint | 零成本启动实验 | 低（GitHub 开源） |
| **P0** | DINOv2 ViT-B/14 权重 | 核心奖励信号 | 低（HuggingFace） |
| **P1** | RAFT 光流模型 | Dynamic Degree 检测 | 低（开源） |
| **P1** | 单张 GPU（RTX 3090/4090 级别） | DIAMOND 训练 | 取决于现有硬件 |
| **P2** | Oasis 500M checkpoint | 泛化验证 | 低（开源） |
| **P2** | 多卡 GPU 集群 | Wan2.1 大模型实验 | 高（需申请） |
| **P3** | HPSv3 / DepthAnythingV3 | 辅助奖励 | 低（开源） |
