# 新手上路

在本节中，我们将简要叙述与编译器有关的相关基本知识与概念，并介绍 LLVM 项目。

???+note "Note"
    本节假设您没有任何与编译器相关的背景且假设您已对编程有所了解，可编写简单的 C 代码。

## 什么是编译器

从概念上讲，编译器是一种将一种编程语言转换为另一种语言的**计算机程序**。而我们通常所说的编译器，一般会将高阶编程语言如 C, C++, Fortran 等转换为更底层的汇编语言或者机器代码。

**注意编译器不是编辑器，更不是 IDE。**

## LLVM

???+note "Note"
    注意这里讨论的 LLVM 指包含在 LLVM monorepo 中的所有子项目，而不是指传统意义上的 LLVM 核心。

LLVM 是一组**可重用的**编译器工具链。目前它至少包含以下几个子项目：

* bolt

一个链接后的优化器，通过基于采样剖析器（如Linux perf工具）所收集的执行概况来优化应用程序的代码布局，从而实现改进。

* clang

一个与 GCC 兼容的 C 家族（C, C++, Objective C/C++, OpenCL, CUDA）编译器前端，以可观的编译时间与良好的诊断信息而著名。

* libcxx

C++ 标准库的一个新实现，针对 C++11 及以上版本。


* llvm

LLVM 项目的核心，包括 LLVM 中间表示，各种优化及其对应的命令行工具。

* mlir

一个用于编写编译器中各类分析与变换及其所需的中间表示的通用框架。

* lld

一个高性能，模块化的跨平台链接器。

## LLVM 的设计理念

LLVM 是模块化的。 LLVM 最重要的方面是它被设计为一组库，而不是像 GCC 那样的单一命令行编译器或像 JVM 或 .NET 虚拟机那样的不透明虚拟机。

举个例子， LLVM 优化器提供了大量 [pass](../pass/index.zh.md)，它们被编译成一个或多个 `.o` 文件，然后被构建成一系列静态或动态库。这些库提供各种分析和转换功能，并且 pass 之间尽可能保持独立，或者如果它们依赖于其他分析来完成其工作，则明确声明它们与其他 pass 之间的依赖关系。而在 LLVM 优化实际的优化过程中，只有被使用的 pass 会被链接到最终应用程序，而不是所有的。

这种直接的设计方法允许 LLVM 提供大量的功能，其中一些可能只对特定的受众有用，而不会影响到只想做其他简单事情的库的用户。相比之下，传统的编译器优化器是作为大量相互关联的代码构建的，很难对其进行子集化、推理和加速。使用 LLVM，您可以在不知道整个系统如何组合在一起的情况下了解各个优化器。

## 编译器的流程（以 Clang/LLVM 为例）

通常来说，编译器分为前端，中端与后端。

前端的职责主要在于对源代码进行分析和处理并生成中间代码。

### 词法分析

在这个阶段，编译器会将源代码分成一个个的 token，并忽略掉诸如空格空行等无效信息。每个 token 通常包含以下信息：

* 类型。常见的类型如标识符、关键词、分隔符、运算符、字面量等。
* 值。此 token 实际对应的值。
* 源代码中的位置。记录 token 在原始代码中的位置可帮助我们生成更好的诊断信息。

举个例子:
```c
int foo(int i) {
    return i + 42;
}

int main() {
    int x = foo(24);
    return x - 24;
}
```
对于以上代码，Clang 会生成如下 token:

```sh
$ clang -c -Xclang -dump-tokens prog.c
int 'int'        [StartOfLine]  Loc=<prog.c:1:1>
identifier 'foo'         [LeadingSpace] Loc=<prog.c:1:5>
l_paren '('             Loc=<prog.c:1:8>
int 'int'               Loc=<prog.c:1:9>
identifier 'i'   [LeadingSpace] Loc=<prog.c:1:13>
r_paren ')'             Loc=<prog.c:1:14>
l_brace '{'      [LeadingSpace] Loc=<prog.c:1:16>
return 'return'  [LeadingSpace] Loc=<prog.c:1:18>
identifier 'i'   [LeadingSpace] Loc=<prog.c:1:25>
plus '+'         [LeadingSpace] Loc=<prog.c:1:27>
numeric_constant '42'    [LeadingSpace] Loc=<prog.c:1:29>
semi ';'                Loc=<prog.c:1:31>
r_brace '}'      [LeadingSpace] Loc=<prog.c:1:33>
int 'int'        [StartOfLine]  Loc=<prog.c:3:1>
identifier 'main'        [LeadingSpace] Loc=<prog.c:3:5>
l_paren '('             Loc=<prog.c:3:9>
r_paren ')'             Loc=<prog.c:3:10>
l_brace '{'      [LeadingSpace] Loc=<prog.c:3:12>
int 'int'        [StartOfLine] [LeadingSpace]   Loc=<prog.c:4:3>
identifier 'x'   [LeadingSpace] Loc=<prog.c:4:7>
equal '='        [LeadingSpace] Loc=<prog.c:4:9>
identifier 'foo'         [LeadingSpace] Loc=<prog.c:4:11>
l_paren '('             Loc=<prog.c:4:14>
numeric_constant '24'           Loc=<prog.c:4:15>
r_paren ')'             Loc=<prog.c:4:17>
semi ';'                Loc=<prog.c:4:18>
return 'return'  [StartOfLine] [LeadingSpace]   Loc=<prog.c:5:3>
identifier 'x'   [LeadingSpace] Loc=<prog.c:5:10>
minus '-'        [LeadingSpace] Loc=<prog.c:5:12>
numeric_constant '24'    [LeadingSpace] Loc=<prog.c:5:14>
semi ';'                Loc=<prog.c:5:16>
r_brace '}'      [StartOfLine]  Loc=<prog.c:6:1>
eof ''          Loc=<prog.c:6:2>
```

### 语法分析

在这个阶段，编译器会将线性的 token 流转换为具有特定语法规则的抽象语法树（Abstract Syntax Tree）。如果在语法解析中出现错误，编译器也会在这个阶段抛出一个诊断。

对于前面的代码，可以用以下命令查看 Clang 所生成的语法树：
```sh
$ clang -Xclang -ast-dump -fsyntax-only prog.c
TranslationUnitDecl 0x820688 <<invalid sloc>> <invalid sloc>
|-FunctionDecl 0x877270 <prog.c:1:1, col:33> col:5 used foo 'int (int)'
| |-ParmVarDecl 0x8771a0 <col:9, col:13> col:13 used i 'int'
| `-CompoundStmt 0x8773e8 <col:16, col:33>
|   `-ReturnStmt 0x8773d8 <col:18, col:29>
|     `-BinaryOperator 0x8773b8 <col:25, col:29> 'int' '+'
|       |-ImplicitCastExpr 0x8773a0 <col:25> 'int' <LValueToRValue>
|       | `-DeclRefExpr 0x877360 <col:25> 'int' lvalue ParmVar 0x8771a0 'i' 'int'
|       `-IntegerLiteral 0x877380 <col:29> 'int' 42
`-FunctionDecl 0x877450 <line:3:1, line:6:1> line:3:5 main 'int ()'
  `-CompoundStmt 0x8776b8 <col:12, line:6:1>
    |-DeclStmt 0x877618 <line:4:3, col:18>
    | `-VarDecl 0x877508 <col:3, col:17> col:7 used x 'int' cinit
    |   `-CallExpr 0x8775f0 <col:11, col:17> 'int'
    |     |-ImplicitCastExpr 0x8775d8 <col:11> 'int (*)(int)' <FunctionToPointerDecay>
    |     | `-DeclRefExpr 0x877570 <col:11> 'int (int)' Function 0x877270 'foo' 'int (int)'
    |     `-IntegerLiteral 0x877590 <col:15> 'int' 24
    `-ReturnStmt 0x8776a8 <line:5:3, col:14>
      `-BinaryOperator 0x877688 <col:10, col:14> 'int' '-'
        |-ImplicitCastExpr 0x877670 <col:10> 'int' <LValueToRValue>
        | `-DeclRefExpr 0x877630 <col:10> 'int' lvalue Var 0x877508 'x' 'int'
        `-IntegerLiteral 0x877650 <col:14> 'int' 24
```
### 中间代码生成

在这个阶段，编译器会将抽象语法树转换为中间表示。

还是上面的例子，Clang 会生成以下中间表示（LLVM IR）：
```llvm
; ModuleID = 'prog.c'
source_filename = "prog.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: noinline nounwind optnone
define dso_local i32 @foo(i32 noundef %i) #0 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32, i32* %i.addr, align 4
  %add = add nsw i32 %0, 42
  ret i32 %add
}

; Function Attrs: noinline nounwind optnone
define dso_local i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %x = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  %call = call i32 @foo(i32 noundef 24)
  store i32 %call, i32* %x, align 4
  %0 = load i32, i32* %x, align 4
  %sub = sub nsw i32 %0, 24
  ret i32 %sub
}
```

中端主要负责在中间代码上进行分析和优化，生成更高效的中间代码。除了通用的优化，中端也包含了那些平台相关的特殊优化。

中端的具体职责包含：

* 对中间表示进行分析，如数据流分析，别名分析等。此阶段是编译器进行其他优化的基础。
* 优化中间表示，将其表述被转化为功能等同但更快（或更小）的形式。常见的优化如内联展开，死代码消除，常量传播、循环转换等。

例如我们有这么一段 LLVM IR:
```llvm
$ cat prog.ll
define void @test(ptr %Q, ptr %P) {
  %DEAD = load i32, ptr %Q
  store i32 %DEAD, ptr %P
  store i32 0, ptr %P
  ret void
}
```

我们可以使用 LLVM 提供的命令行工具对其进行一段任意的优化。通过配合 Unix 系统上的管道符，我们可以非常灵活地达到各种效果：
```sh
$ opt -passes=dse  prog.ll | llvm-dis -f
; ModuleID = '<stdin>'
source_filename = "prog.ll"

define void @test(ptr %Q, ptr %P) {
  store i32 0, ptr %P, align 4
  ret void
}
```

这里我们使用了 LLVM 提供的 `opt` 工具单独对 `prog.ll` 使用了死代码消除的优化。由于 `opt` 会生成人类不可阅读的 LLVM 字节码，我们接着又使用了 `llvm-dis` 将其转换为 LLVM IR。

后端需要将中间代码生成对应平台的机器代码，以及进行一些体系架构的细节的优化。

## 如何给 LLVM 做贡献

### 提交新 Bug

在过去的很长一段时间里，LLVM 一直使用 [Bugzilla](https://bugs.llvm.org/) 作为其 Bug tracker，但是现在已转向 [Github issues](https://github.com/llvm/llvm-project/issues)。您可以通过查看 issue 中的 [Good Frist Issue](https://github.com/llvm/llvm-project/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) 来寻找一些容易上手的 Bug。

### 提交补丁

???+note "Note"
    注意，LLVM 有意向放弃 Phabricator 并转向 Github Pull Requests，详见 [^Code Review Process Update]。

[^CodeReviewProcessUpdate]: 您可以参考 [本帖](https://discourse.llvm.org/t/code-review-process-update/63964) 了解更多信息。

目前，LLVM 还主要使用 [Phabricator](https://reviews.llvm.org/) 来提交补丁进行代码审查。您可以通过使用 Github 等第三方账号快速注册一个 Phabricator 账号。

您可以通过使用网页或 Phabricator 自带的命令行工具 arcanist 来提交您的补丁。具体文档可参考[这里](https://llvm.org/docs/Phabricator.html#phabricator-reviews)。下面是使用 arcanist 的简单示例：

???+note "Note"
    这里我们假设您使用的是 Unix-like 的操作系统并已安装并配置好了 Git

首先我们需要安装 arcanist：
```sh
git clone https://github.com/phacility/arcanist.git
```

接着将其添加到环境变量中：
```sh
export PATH="$PATH:/path/to/arcanist/bin/"
```

授权登陆：
```sh
arc install-certificate
```

在进行编码时，我们推荐使用 `amend` 或手动 `rebase` 将我们的代码压缩成一个 `commit`，方便后续的代码审查。

当我们的 patch 在本地通过测试后，便可以准备发出去了。但在此之前一定要注意用 clang-format 保证我们的 patch 符合 LLVM Coding Style。

建议使用 LLVM 官方提供的 [git-clang-format](https://github.com/llvm/llvm-project/blob/main/clang/tools/clang-format/git-clang-format) 工具，效果最好，避免意外修改到其他的代码。 可以直接将 git-clang-format 所在的路径添加到 PATH 中就可以了。

```sh
git clang-format HEAD~1
```

注意format后并不会自动提交，所以您需要：
```sh
git commit --amend -a
```

在上述工作完成后，我们便可以使用 arc 提交补丁到 Phabricator 了：
```sh
arc diff
```
arcanist 会将最近的一个提交上传到 Phabricator 上，成功后它会给我们一个链接，这便是我们的 Revision 啦。

在 reviewer 看过代码后，可能会给我们一些修改的建议，我们可以继续使用 arcanist 更新我们的补丁:
```sh
arc diff --update DXXXXX
```
其中 DXXXXX 是我们之前的 Revision，因为 arcanist 无法知道上下文，所以我们必须手动指定。

### 与社区沟通

* 邮件列表

尽管社区已逐步淘汰了邮件列表，我们仍建议您订阅对应子项目的提交记录列表，这有助于帮助您了解项目中正在发生的事情。举个例子，如果您想向 Clang 做贡献，您可以订阅 [cfe-commits](https://lists.llvm.org/mailman/listinfo/cfe-commits) 这个列表，如果您想向 LLVM 核心做贡献，可订阅 [llvm-commits](https://lists.llvm.org/mailman/listinfo/llvm-commits) 这个列表。

* [Discourse](https://discourse.llvm.org/)
* [Discord](https://discord.com/invite/xS7Z362)

### 相关链接

* [Getting Started with the LLVM System](https://llvm.org/docs/GettingStarted.html)
* [The Design Decisions of LLVM](https://www.aosabook.org/en/llvm.html)
* [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html)
* [LLVM IR Tutorial - Phis, GEPs and other things, oh my!](https://youtu.be/m8G_S5LwlTo)
* [2019 LLVM Developers’ Meeting: E. Christopher & J. Doerfert “Introduction to LLVM”](https://youtu.be/J5xExRGaIIY)
* [“Clang” CFE Internals Manual](https://clang.llvm.org/docs/InternalsManual.html)
* [2019 LLVM Developers’ Meeting: S. Haastregt & A. Stulova “An overview of Clang ”](https://youtu.be/5kkMpJpIGYU)
* [P. Goldsborough “clang-useful: Building useful tools with LLVM and clang for fun and profit"](https://youtu.be/E6i8jmiy8MY)


