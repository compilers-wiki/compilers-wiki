# Multi-Level Intermediate Representation

**MLIR 项目正在极速发展中，情况随时有可能发生变化，请在阅读时关注页面的书写时间**.

MLIR 项目是一次全新的，构建可重用、可扩展的编译器基础设施的尝试。它旨在解决软件碎片化问题，改进异构硬件上的编译，减少领域特定编译器的开发代价，以及连接现有的编译器设施。

简单来说，MLIR 是用于编写编译器中各类分析及其所需的中间表示的通用框架。其提供了一系列设施以供编译器开发者方便快速地定义新的 IR 与使用现有的各类 IR. 开发者可以方便地混用各类 IR, 在它们之中与之间自由转换，进行不同层次的分析，最终生成目标代码。

MLIR 项目中还包含了一些已经编写好的 "IR", 可供直接使用。可以从 MLIR 的基本概念与组成模块：[方言](basic/dialect.zh.md) 开始了解这些现有的 "IR" 与学习如何构建自己的 "IR".

## 本 Wiki 简介

本 Wiki 旨在：

- 辅助 MLIR 本身的开发
- 辅助基于 MLIR 的新方言的开发
- 辅助对 MLIR 自带方言与 pass 的使用

## 相关链接

- [官方文档](https://mlir.llvm.org/docs/)
- [CodeGen Dialect Overview](https://discourse.llvm.org/t/codegen-dialect-overview/2723)
    - 一篇介绍 MLIR 中各个方言下降路径的帖子，虽然老了一点，有些方言已经被拆了，但是其中的分类值得深思。其中的图片更是**一图胜千言，不看字也得看图。**
    - 作者将 MLIR 按照数据 - 算法两个维度分类。数据从 Tensor -> Buffer, 逐渐从抽象的张量具体到内存。算法从 Structure -> Payload, 也就是从高层的，结构化的描述到底层的，命令式的东西
    - 作者还分享了各个方言的关注点
- [如何评价 MLIR 项目中 Linalg Dialect 的设计思想？ - mackler 的回答 - 知乎 ](https://www.zhihu.com/question/442964082/answer/1718438248)
    - 对 MLIR 的比较 "尖锐" 的意见
    - 其对编译器优化的分类令人眼前一亮：auto-tiling/auto-tensorize/auto-schedule
- 一些关注 MLIR 的中文使用者，可以跟着 timeline 看看
    - [hunterzju](https://www.zhihu.com/people/hunterzju-45): 主要关注
    - [mackler](https://www.zhihu.com/people/mackler): 主要偏向 MLIR 在 AI 领域的工作，虽然 ta 可能更多的是关注广义的计算机体系和优化
    - [法斯特豪斯](https://www.zhihu.com/people/zhang-hong-bin-99): Buddy-mlir 项目的 "主持人", 基本上主页都是关于 MLIR 的信息
    - [MLIR 中国社区](https://www.zhihu.com/people/mlir-70): 顾名思义
