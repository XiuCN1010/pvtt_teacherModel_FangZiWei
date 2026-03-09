# TeacherModel_FangZiWei
# E-commerce Video Generation & Editing (Wan2.1 + VACE)

本项目致力于探索基于 **Wan2.1** 与 **VACE** 的视频生成与编辑技术，核心目标是**为电商平台提供高效的广告视频生成方案**。通过将商品参考图精准、高质量地嵌入目标视频，实现低成本、高效率的电商素材生产。

---

## 项目核心愿景
* **高效替换：** 实现用户输入图片到目标视频的无缝转换。
* **结构保持：** 确保生成视频能够完美保留电商商品的原始结构与细节质量。
* **落地场景：** 自动化生成电商短视频广告，助力商家快速出片。

---

## Weekly Schedule & Progress

### Week 1: 熟悉项目背景及已有研究进展
*重点：梳理核心技术栈，确立以“图片 $\rightarrow$ 视频”为核心的电商生成链路。*

* ** 核心技术调研 (Core Research)**
    * [x] 阅读 **Wan2.1** 技术报告：*Open and Advanced Large-Scale Video Generative Models*。
    * [x] **项目定位梳理：** 确定将 AI 用于电商广告生成（参考图切换至目标视频），关注高效替换与结构稳定性。
* ** 基础知识补全 (Knowledge)**
    * [x] 阅读 `global-optima-research` 相关仓库。
    * [x] 研读 **DiT** 论文：*Scalable Diffusion Models with Transformers*。

---

### Week 2: 深入学习 Wan 与 VACE 模型
*重点：模型复现、微调流程验证及新方案探索。*

* ** 核心技术调研 (Core Research)**
    * [x] **Wan2.1 深入学习：** 结合技术报告与源码分析架构
    * [x] **新方向探索：** 调研 **StructXLIP** 表现，探讨其在保持商品结构特征方面的优势，评估其接入优化工序的可行性。
* ** 代码工程 (Engineering)**
    * [x] 跑通 Wan2.1 T2V 的推理过程
    * [x] 在 5090 Server 中配置 **DiffSynth Studio** 环境。
* ** 基础知识补全 (Knowledge)**
    * [x] 阅读 **Flow Matching** 与 **Rectified Flow** 论文，理解主流生成模型数学基础。

---

## Week 3: 深入了解微调 与StruXLIP框架
*重点：模型复现、微调流程验证及新方案探索。*
1.  [x] **StructXLIP 集成实验：** 通过文本结构化过滤提升模型处理丰富视觉结构图像的效果（边缘图和词表过滤），可整合至微调框架中，无推理成本。
2.  [x] **Wan2.2 的配置与模型选择：** Wan2.2-Animate-14B 和 PAI/Wan2.2-VACE-Fun-A14B（阿里团队）
3.  [x] **数据选择：** DAVIS
4.  [ ] **测试报告：** 测试Wan2.2模型在LoRA微调后的视频质量， 并分析可优化点。
