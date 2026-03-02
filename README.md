# TeacherModel_FangZiWei
# 🚀 E-commerce Video Generation & Editing (Wan2.1 + VACE)

本项目致力于探索基于 **Wan2.1** 与 **VACE** 的视频生成与编辑技术，核心目标是**为电商平台提供高效的广告视频生成方案**。通过将商品参考图精准、高质量地嵌入目标视频，实现低成本、高效率的电商素材生产。

---

## 🎯 项目核心愿景
* **高效替换：** 实现用户输入图片到目标视频的无缝转换。
* **结构保持：** 确保生成视频能够完美保留电商商品的原始结构与细节质量。
* **落地场景：** 自动化生成电商短视频广告，助力商家快速出片。

---

## 📅 Weekly Schedule & Progress

### Week 1: 熟悉项目背景及已有研究进展
*重点：梳理核心技术栈，确立以“图片 $\rightarrow$ 视频”为核心的电商生成链路。*

* ** 核心技术调研 (Core Research)**
    * [x] 阅读 **Wan2.1** 技术报告：*Open and Advanced Large-Scale Video Generative Models*。
    * [x] **项目定位梳理：** 确定将 AI 用于电商广告生成（参考图切换至目标视频），关注高效替换与结构稳定性。
* ** 代码工程 (Engineering)**
    * [x] 在 SuperPod 中配置 Wan2.1 开发环境。
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

## 📈 未来规划 (Next Steps)

1.  **StructXLIP 集成实验：** 验证 StructXLIP 在 I2V (Image-to-Video) 任务中对复杂物体结构的编码精度。
2.  **电商 Pipeline 构建：** 结合 Wan2.1 与微调后的 LoRA，搭建一套自动化的“商品图 -> 广告视频”工作流。
3.  **性能优化：** 探索如何在保证质量的前提下，进一步提升视频生成的推理速度。
