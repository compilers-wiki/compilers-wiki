# 编写一个 Hello World Pass

在本节中，我们将会带领您编写并运行一个非常简单的 pass，这个 pass 可以输出在输入的 IR 模块中包含的所有函数的名称。

## 设置构建环境

假设您已经克隆了 [LLVM 仓库]并在 `build` 目录下设置好了构建树[^how2build]，并且当前工作目录是 LLVM 仓库的根目录。本节中我们将在树内构建新的 pass，这意味着新的 pass 的所有代码都包含在 LLVM 仓库的源代码目录下。

[LLVM 仓库]: https://github.com/llvm/llvm-project
[^how2build]: 请参阅 [LLVM 官方文档](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm)了解如何从源码构建 LLVM。

在 `llvm/include/llvm/Transforms/Utils/` 下创建一个名为 `Hello.h` 的文件：

```sh
touch llvm/include/llvm/Transforms/Utils/Hello.h
```

在 `llvm/lib/Transforms/Utils/` 下创建一个名为 `Hello.cpp` 的文件：

```sh
touch llvm/lib/Transforms/Utils/Hello.cpp
```

修改 `llvm/lib/Transforms/Utils/CMakeLists.txt` 并将 `Hello.cpp` 加入到对 `add_llvm_component_library` 的调用中：

```CMake
add_llvm_component_library(LLVMTransformUtils
    # Other files ...
    Hello.cpp
    # Other files and configurations ...
)
```

这一步是为了将 `Hello.cpp` 添加到 LLVM 的构建过程中。

## 代码实现

将如下内容添加到 `Hello.h` 中：

```cpp
#ifndef LLVM_TRANSFORMS_HELLO_H
#define LLVM_TRANSFORMS_HELLO_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class HelloPass : public PassInfoMixin<HelloPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

}  // namespace llvm

#endif
```

将如下内容添加到 `Hello.cpp` 中:

```cpp
#include "llvm/Transforms/Utils/Hello.h"

namespace llvm {

PreservedAnalyses HelloPass::run(Function &F, FunctionAnalysisManager &AM) {
  errs() << F.getName() << "\n";
  return PreservedAnalyses::all();
}

}  // namespace llvm
```

是的！这就是（几乎）全部为了实现一个简单的 pass 您需要编写的代码。让我们将这段代码进一步分解并看看它是如何工作的。

为了使用 C++ 创建一个 pass，您需要编写一个实现了 pass 所需要的软件接口的类。传统的做法是从某个基类派生出来一个类并重写某些虚函数以实现接口，但新的 pass 管理器并没有采用这种方法，而是采用了一种被称为 _基于概念的多态_（Concept-Based Polymorphism）[^concepts-based-polymorphism]的方法。**只要这个类包含一个名为 `run` 的方法使之可以在一段 IR 上执行，那么这个类就表示一个 pass。**不需要从某个基类派生出子类来表示一个 pass。在上面的代码中，我们的 `HelloPass` 类继承自 `PassInfoMixin` 类，这个基类向 `HelloPass` 提供的接口中又额外添加了一些通用的代码。但真正重要的部分是 `run` 函数，正是 `run` 函数的存在使得 `HelloPass` 表示一个 pass。

[^concepts-based-polymorphism]: [这里](https://gist.github.com/GuillaumeDua/b0f5e3a40ce49468607dd62f7b7809b1)有一份对基于概念的多态的详细介绍和讨论文档。

`run` 函数接收两个参数。第一个参数 `F` 是一个 IR 函数，新的 pass 将在这个函数上运行。注意 `F` 是通过一个非常量引用传递进来的，这说明我们可以修改这个函数（也就是对这个函数中包含的 IR 进行修改和变换）。第二个参数 `AM` 是一个 pass 管理器的实例；通过这个实例，新的 pass 可以访问各种分析 pass 并获得函数级别的分析信息。

由于 `run` 函数将 `Function` 作为参数，因此 `HelloPass` 是一个**函数 pass**。Pass 管理器会安排函数 pass 依次在输入的 IR 模块中包含的每个函数上执行。当 `HelloPass` 在某个 IR 函数上执行时，它会将函数的名称写入标准错误流然后退出。

`run` 函数返回一个 `PreservedAnalyses` 对象。这个对象包含一些信息用于指示当当前的 pass 运行完毕后，已有的分析信息哪些是可以继续使用的，哪些是不能继续使用的。`HelloPass` 的 `run` 函数返回 `PreservedAnalyses::all()` 来指示所有的分析信息都继续可用（这是因为 `HelloPass` 并没有对 IR 进行修改，也就不会破坏任何已有的分析信息）。

## 注册 Pass

我们已经完成了对这个简单的 pass 的实现，但是我们还没有告诉 pass 管理器新的 pass 的存在。我们需要向 pass 管理器 _注册_ 新的 pass。

编辑 `llvm/lib/Passes/PassRegistry.def` 并向其中添加如下内容：

```cpp
FUNCTION_PASS("hello", HelloPass())
```

注意 `FUNCTION_PASS` 的第一个参数就是新的 pass 的名称。

向 `llvm/lib/Passes/PassBuilder.cpp` 中添加如下的 `#include`：

```cpp
#include "llvm/Transforms/Utils/Hello.h"
```

注册工作就完成了。接下来我们将构建并运行新的 pass。

## 构建并运行

进入构建目录并构建 `opt`。`opt` 是一个专门用于在一段 IR 上运行一个或一系列 pass 的工具。当 `opt` 构建完毕后，创建一个 `test.ll` 文件用于测试：

```llvm
define i32 @foo() {
  %a = add i32 2, 3
  ret i32 %a
}

define void @bar() {
  ret void
}
```

然后使用 `opt` 工具运行新的 pass：

```sh
build/bin/opt -disable-output test.ll -passes=hello
```

`-passes=hello` 编译选项命令 `opt` 运行 `HelloPass`。

`opt` 应该输出如下内容：

```
foo
bar
```

恭喜！您完成了您的 LLVM pass！
