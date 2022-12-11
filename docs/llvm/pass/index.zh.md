# Pass

LLVM Pass 框架是 LLVM 的一个重要组成部分，编译器的大量核心代码都包含在 LLVM pass 中。LLVM pass 可以对程序进行变换和优化，可以对程序进行分析并输出变换和优化所必须的分析信息；除此之外，LLVM pass 也是一种对编译器代码进行组织的工具。

一个 LLVM pass 可以被视为一个以 LLVM IR 作为输入、并产生某种依赖于具体 pass 的输出的黑盒。有两种不同的 pass：**变换 pass**（Transformation Pass） 以及**分析 pass**（Analysis Pass）。一个变换 pass 通常会修改输入的 IR 并将转换后的 IR 作为输出。另外，将输入的 LLVM IR 下降到 SelectionDAG 以及机器 IR 的 pass 也是变换 pass。一个分析 pass 不会修改输入的 IR，但它会对输入的 IR 进行分析并产生描述 IR 的某种属性的信息。变换 pass 通常用于实现编译优化、程序变换、程序插桩等任务。分析 pass 通常用于为变换 pass 提供必要的信息。

真实世界的编译器通常会依赖于上百个在 IR 上依次执行的 pass 来完成程序优化以及代码生成工作。这一个由多个 pass 组成的、每个 pass 依次执行的 pass 序列一般也被称为**管线（The Pipeline）**。需要注意，不同的编译配置通常会影响到管线的配置，也就是管线中使用的 pass 集合以及 pass 执行的先后顺序。

Pass 可以有**依赖**。例如，一个变换 pass 必须在它所有依赖的分析 pass 完成之后才能正常运行。另外，某些变换 pass 会破坏某些已经得到并缓存下来的代码分析结果。LLVM 使用**pass 管理器**（Pass Manager）对可用的 pass 进行管理并将 pass 管线调度到输入的 IR 上执行。但是，由于历史原因，目前 LLVM 有两组不同的 pass manager：新 pass 管理器（New Pass Manager）以及旧 pass manager（Legacy Pass Manager）。本 wiki 主要介绍新 pass 管理器；LLVM 社区也正在将代码迁移到新 pass 管理器并弃用旧 pass 管理器。
