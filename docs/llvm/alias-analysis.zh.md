## 引言

别名分析 (Alias Analysis)，又叫指针分析 (Pointer Analysis) [^pointer]。这种技术试图确定两个指针是否可以指向内存中的同一个对象。 有许多不同的别名分析算法和许多不同的分类方法：流敏感 (flow-sensitive) 与流不敏感 (flow-insensitive) 、上下文敏感 (context-sensitive) 与上下文不敏感 (context-insensitive) 、字段敏感 (field-sensitive) 与字段不敏感 (field-insensitive) 、基于 Unification [^unification] 与基于 Subset [^subset] 等。一般来说，别名分析以 [Must, May, or No](#Must-May-No) 别名响应响应查询，表明两个指针始终指向同一个对象，可能指向同一个对象 ，或者已知永远不会指向同一个对象。

LLVM [AliasAnalysis](https://llvm.org/doxygen/classllvm_1_1AliasAnalysis.html) 类是客户端使用的主要接口和 LLVM 系统中别名分析的实现。 这个类是别名分析信息的客户端与提供它的实现之间的通用接口，旨在支持广泛的实现和客户端（虽然目前所有客户端都被假定为流不敏感 (flow-insensitive) 的 ）。除了简单的别名分析信息之外。这个类还暴露了与其实现相关的 Mod/Ref 信息，允许后端多种强大的分析和转换能够很好地协同工作。

本文档包含成功实现此接口、使用它以及测试双方所需的信息。它还解释了关于结果的确切含义的一些细节。

[^pointer]: \ 细节上，两者有些许不同
[^unification]: 又称 Steensgaard-style
[^subset]: 又称 Andersen-style


## `AliasAnalysis` 类简介

[AliasAnalysis](https://llvm.org/doxygen/classllvm_1_1AliasAnalysis.html) 类定义了各种别名分析实现应支持的接口。 此类导出两个重要的 enum：`AliasResult` 和 `ModRefResult`，分别表示别名查询或 mod/ref 查询的结果。

`AliasAnalysis` 接口以几种不同的方式表示暴露了内存相关信息。具体来说，我们将内存对象表示为起始地址和大小，函数调用表示为执行调用的实际 `call` 或 `invoke` 指令。 `AliasAnalysis` 接口还暴露了一些获取任意指令的 mod/ref 信息辅助方法。

所有 `AliasAnalysis` 接口都要求在涉及多个 `Value` 的查询中，不是常量的值都在同一个函数中定义。

### 指针的表示方法

最重要的是，`AliasAnalysis` 类提供了几种方法，用于查询两个内存对象是否别名，函数调用是否可以修改或读取内存对象等。对于所有这些查询，内存对象表示为一对 它们的起始地址（LLVM `Value*`）和静态大小。

将内存对象表示为起始地址和大小对于正确的别名分析至关重要。 例如，考虑这个（有点难看，但我们也可能遇到）C 代码:

``` c++
int i;
char C[2];
char A[10];
/* ... */
for (i = 0; i != 10; ++i) {
  C[0] = A[i];          /* One byte store */
  C[1] = A[9-i];        /* One byte store */
}
```

在这种情况下，`basic-aa` pass 将消除对 `C[0]`和`C[1]` 的存储的歧义，因为它们是对相隔一个字节的两个不同位置的访问，并且每个访问都是一个字节。 在这种情况下，`licm` pass 可以将循环内部的 store 语句提升到循环外部。另一方面，有些代码无法消歧义，例如：

``` c++
int i;
char C[2];
char A[10];
/* ... */
for (i = 0; i != 10; ++i) {
  ((short*)C)[0] = A[i];  /* Two byte store! */
  C[1] = A[9-i];          /* One byte store */
}
```

对于这个代码，对 C 的两个 store 互为别名，因为对 `&C[0]` 元素的访问是两个字节的访问。 如果大小信息在查询中不可用，即使是第一种情况也必须保守地假设访问别名。

### `alias` 方法 {#alias}

`alias` 方法是用于确定两个内存对象是否互为别名的主要接口。 它采用两个内存对象作为输入，并根据需要返回 MustAlias、PartialAlias、MayAlias 或 NoAlias。

与所有 `AliasAnalysis` 接口一样，`alias` 方法要求在同一函数中定义两个指针值，或者至少其中一个值是常量。


#### Must, May, and No Alias Responses {#Must-May-No}

???+note "Note"
    *基于*一个指针翻译自英语 "based on the pointer" ，意为一个对象的基地址（开头的地方）。

当*基于*一个指针的任何内存引用与*基于*另一个指针的任何内存对象之间从不存在直接依赖关系时，可以使用 `NoAlias` 响应。大致有如下几种情况：

- 指向不重叠的内存范围
- 仅用于读取内存
- 内存在通过一个指针的访问和通过另一个指针的访问之间被释放和重新分配

中间内存的释放和分配应该自动条件两个指针的数据依赖。


???+note "Irrelevant"
    “Irrelevant” 的依赖项被当成作为一个 `NoAlias` 的特殊情况，因此被忽略。


`MayAlias` 用来表示两个指针可能指向同一个对象。

`PartialAlias` 表示我们已知两个指针可能重叠，他们可能有相同的起始地址，也可能不相同。

`MustAlias` 表明两个地址总是完全重合，`MustAlias` 不保证两个指针相等。

### `getModRefInfo` 方法

`getModRefInfo` 方法返回有关指令的执行是否可以读取或修改内存位置的信息。 Mod/Ref 信息总是保守的：如果一条指令**可能**读取或写入一个位置，则返回 `ModRef`。

`AliasAnalysis` 类还提供了一种用于测试函数调用之间依赖关系的 `getModRefInfo` 方法。 此方法应该传入两个 call site [^callsite]（`CS1` 和 `CS2`），如果两个调用都没有写入内存，则返回 `NoModRef`，如果 `CS1` 读取由 `CS2` 写入的内存，则返回 `Ref`， 如果 `CS1` 写入由 `CS2` 读取或写入的内存，返回 `Mod`，如果 `CS1` 可能读取或写入由 `CS2` 写入的内存，返回 `ModRef`。 请注意，此关系不可交换。

| 返回值 | 语义 |
| :----: | :----: |
| `Ref` | `CS1` 读取由 `CS2` 写入的内存 |
| `Mod` | `CS1` 写入由 `CS2` 读取或写入的内存 |
| `ModRef` |  `CS1` 可能读取或写入由 `CS2` 写入的内存 |


[^callsite]: \ 发生函数调用的地方，在编译器内部作为一个实例，LLVM 中使用 [CallBase](https://llvm.org/doxygen/classllvm_1_1CallBase.html) 类来表示


### 其他实用的 `AliasAnalysis` 方法

其他一些信息通常由各种别名分析实现收集，可以被各种客户端很好地利用。


#### `getModRefInfoMask` 方法

`getModRefInfoMask` 方法基于关于指针是指向全局不变内存 （globally-constant，返回`NoModRef`）还是局部不变内存（locally-invariant） 它返回 `Ref`）返回所提供指针的 Mod/Ref 信息的**界限**。 全局常量内存包括函数、常量全局变量和空指针。 局部不变内存是我们知道在其 SSA 值的生命周期内不变的内存，但不一定在程序的生命周期内不变。例如，`readonly` `noalias` 参数对应的内存，在函数调用时是已知的不变量。 

给定内存位置 `Loc` 的 Mod/Ref 信息 `MRI`，可以使用 `MRI &= AA.getModRefInfoMas(Loc);`等语句对`MRI`进行细化。 另一个比较常见的用法是 `isModSet(AA.getModRefInfoMask(Loc))` 可以用来检查是否可以修改给定位置。 为了方便起见，还有一个方法 `pointsToConstantMemory(Loc)` 与 `isNoModRef(AA.getModRefInfoMask(Loc))` 同义。

#### `doesNotAccessMemory` 和 `onlyReadsMemory`

这些方法用于为函数调用提供非常简单的 mod/ref 信息。 如果分析可以证明函数从不读取或写入内存，或者函数仅从常量内存读取，则 `doesNotAccessMemory` 方法会为函数返回 true。 具有此属性的函数没有副作用，仅依赖于它们的输入参数，如果它们形成公共子表达式或被提升到循环之外，则允许它们被消除。 许多常见函数是这种类型（例如，`sin` 和 `cos`）， 也很有一部分函数不是（例如，修改 `errno` 变量的 `acos`）。

如果分析可以证明（至多）该函数仅从标记为非 `volatile` 的内存中读取，则 `onlyReadsMemory` 方法会为该函数返回 `true` 。 具有此属性的函数没有副作用，仅取决于它们的输入参数和调用时的内存状态。 因为没有更改内存内容的存储指令，这一性质可以消除和移动对这些函数的调用。 

???+note "Note"
    所有满足 `doesNotAccessMemory` 方法的函数也满足 `onlyReadsMemory`。

## 实现一个新的别名分析算法

为 LLVM 编写新的别名分析实现非常简单。 目前，仓库中已经有几个实现可以用作示例参考[^aa-example]。后文提到的信息可以帮助你完成一些细节问题。 

[^aa-example]: \ https://llvm.org/docs/AliasAnalysis.html#various-alias-analysis-implementations

### 不同风格的 Pass

第一步是确定要使用那种类型的 [LLVM Pass](/llvm/pass/index.zh/) 。与大多数其他分析和转换的情况一样，您要解决的问题类型来看，答案应该是比较明显的：

| 需求 | 选用的 Pass 类型 |
| :---: | :---: |
| 跨过程分析 | Pass |
| 函数内部的分析 | FunctionPass |
| 仅仅提供信息，不需要运行 | ImmutablePass |


In addition to the pass that you subclass, you should also inherit from the `AliasAnalysis` interface, of course, and use the `RegisterAnalysisGroup` template to register as an implementation of `AliasAnalysis`.

### Required initialization calls

Your subclass of `AliasAnalysis` is required to invoke two methods on the `AliasAnalysis` base class: `getAnalysisUsage` and `InitializeAliasAnalysis`. In particular, your implementation of `getAnalysisUsage` should explicitly call into the `AliasAnalysis::getAnalysisUsage` method in addition to doing any declaring any pass dependencies your pass has. Thus you should have something like this:

``` c++
void getAnalysisUsage(AnalysisUsage &AU) const {
  AliasAnalysis::getAnalysisUsage(AU);
  // declare your dependencies here.
}
```

Additionally, your must invoke the `InitializeAliasAnalysis` method from your analysis run method (`run` for a `Pass`, `runOnFunction` for a `FunctionPass`, or `InitializePass` for an `ImmutablePass`). For example (as part of a `Pass`):

``` c++
bool run(Module &M) {
  InitializeAliasAnalysis(this);
  // Perform analysis here...
  return false;
}
```

### Required methods to override

You must override the `getAdjustedAnalysisPointer` method on all subclasses of `AliasAnalysis`. An example implementation of this method would look like:

``` c++
void *getAdjustedAnalysisPointer(const void* ID) override {
  if (ID == &AliasAnalysis::ID)
    return (AliasAnalysis*)this;
  return this;
}
```

### Interfaces which may be specified

All of the [AliasAnalysis](https://llvm.org/doxygen/classllvm_1_1AliasAnalysis.html) virtual methods default to providing `chaining <aliasanalysis-chaining>`{.interpreted-text role="ref"} to another alias analysis implementation, which ends up returning conservatively correct information (returning "May" Alias and "Mod/Ref" for alias and mod/ref queries respectively). Depending on the capabilities of the analysis you are implementing, you just override the interfaces you can improve.

### `AliasAnalysis` chaining behavior {#aliasanalysis-chaining}

Every alias analysis pass chains to another alias analysis implementation (for example, the user can specify "`-basic-aa -ds-aa -licm`" to get the maximum benefit from both alias analyses). The alias analysis class automatically takes care of most of this for methods that you don't override. For methods that you do override, in code paths that return a conservative MayAlias or Mod/Ref result, simply return whatever the superclass computes. For example:

``` c++
AliasResult alias(const Value *V1, unsigned V1Size,
                  const Value *V2, unsigned V2Size) {
  if (...)
    return NoAlias;
  ...

  // Couldn't determine a must or no-alias result.
  return AliasAnalysis::alias(V1, V1Size, V2, V2Size);
}
```

In addition to analysis queries, you must make sure to unconditionally pass LLVM [update notification](#update notification) methods to the superclass as well if you override them, which allows all alias analyses in a change to be updated.

### Updating analysis results for transformations {#update notification}

Alias analysis information is initially computed for a static snapshot of the program, but clients will use this information to make transformations to the code. All but the most trivial forms of alias analysis will need to have their analysis results updated to reflect the changes made by these transformations.

The `AliasAnalysis` interface exposes four methods which are used to communicate program changes from the clients to the analysis implementations. Various alias analysis implementations should use these methods to ensure that their internal data structures are kept up-to-date as the program changes (for example, when an instruction is deleted), and clients of alias analysis must be sure to call these interfaces appropriately.

#### The `deleteValue` method

The `deleteValue` method is called by transformations when they remove an instruction or any other value from the program (including values that do not use pointers). Typically alias analyses keep data structures that have entries for each value in the program. When this method is called, they should remove any entries for the specified value, if they exist.

#### The `copyValue` method

The `copyValue` method is used when a new value is introduced into the program. There is no way to introduce a value into the program that did not exist before (this doesn't make sense for a safe compiler transformation), so this is the only way to introduce a new value. This method indicates that the new value has exactly the same properties as the value being copied.

#### The `replaceWithNewValue` method

This method is a simple helper method that is provided to make clients easier to use. It is implemented by copying the old analysis information to the new value, then deleting the old value. This method cannot be overridden by alias analysis implementations.

#### The `addEscapingUse` method

The `addEscapingUse` method is used when the uses of a pointer value have changed in ways that may invalidate precomputed analysis information. Implementations may either use this callback to provide conservative responses for points whose uses have change since analysis time, or may recompute some or all of their internal state to continue providing accurate responses.

In general, any new use of a pointer value is considered an escaping use, and must be reported through this callback, *except* for the uses below:

- A `bitcast` or `getelementptr` of the pointer
- A `store` through the pointer (but not a `store` *of* the pointer)
- A `load` through the pointer

### Efficiency Issues

From the LLVM perspective, the only thing you need to do to provide an efficient alias analysis is to make sure that alias analysis **queries** are serviced quickly. The actual calculation of the alias analysis results (the "run" method) is only performed once, but many (perhaps duplicate) queries may be performed. Because of this, try to move as much computation to the run method as possible (within reason).

### Limitations

The AliasAnalysis infrastructure has several limitations which make writing a new `AliasAnalysis` implementation difficult.

There is no way to override the default alias analysis. It would be very useful to be able to do something like "`opt -my-aa -O2`" and have it use `-my-aa` for all passes which need AliasAnalysis, but there is currently no support for that, short of changing the source code and recompiling. Similarly, there is also no way of setting a chain of analyses as the default.

There is no way for transform passes to declare that they preserve `AliasAnalysis` implementations. The `AliasAnalysis` interface includes `deleteValue` and `copyValue` methods which are intended to allow a pass to keep an AliasAnalysis consistent, however there's no way for a pass to declare in its `getAnalysisUsage` that it does so. Some passes attempt to use `AU.addPreserved<AliasAnalysis>`, however this doesn't actually have any effect.

Similarly, the `opt -p` option introduces `ModulePass` passes between each pass, which prevents the use of `FunctionPass` alias analysis passes.

The `AliasAnalysis` API does have functions for notifying implementations when values are deleted or copied, however these aren't sufficient. There are many other ways that LLVM IR can be modified which could be relevant to `AliasAnalysis` implementations which can not be expressed.

The `AliasAnalysisDebugger` utility seems to suggest that `AliasAnalysis` implementations can expect that they will be informed of any relevant `Value` before it appears in an alias query. However, popular clients such as `GVN` don't support this, and are known to trigger errors when run with the `AliasAnalysisDebugger`.

The `AliasSetTracker` class (which is used by `LICM`) makes a non-deterministic number of alias queries. This can cause debugging techniques involving pausing execution after a predetermined number of queries to be unreliable.

Many alias queries can be reformulated in terms of other alias queries. When multiple `AliasAnalysis` queries are chained together, it would make sense to start those queries from the beginning of the chain, with care taken to avoid infinite looping, however currently an implementation which wants to do this can only start such queries from itself.

## Using alias analysis results

There are several different ways to use alias analysis results. In order of preference, these are:

### Using the `MemoryDependenceAnalysis` Pass

The `memdep` pass uses alias analysis to provide high-level dependence information about memory-using instructions. This will tell you which store feeds into a load, for example. It uses caching and other techniques to be efficient, and is used by Dead Store Elimination, GVN, and memcpy optimizations.

### Using the `AliasSetTracker` class {#AliasSetTracker}

Many transformations need information about alias **sets** that are active in some scope, rather than information about pairwise aliasing. The [AliasSetTracker](https://llvm.org/doxygen/classllvm_1_1AliasSetTracker.html) class is used to efficiently build these Alias Sets from the pairwise alias analysis information provided by the `AliasAnalysis` interface.

First you initialize the AliasSetTracker by using the "`add`" methods to add information about various potentially aliasing instructions in the scope you are interested in. Once all of the alias sets are completed, your pass should simply iterate through the constructed alias sets, using the `AliasSetTracker` `begin()`/`end()` methods.

The `AliasSet`s formed by the `AliasSetTracker` are guaranteed to be disjoint, calculate mod/ref information and volatility for the set, and keep track of whether or not all of the pointers in the set are Must aliases. The AliasSetTracker also makes sure that sets are properly folded due to call instructions, and can provide a list of pointers in each set.

As an example user of this, the [Loop Invariant Code Motion](doxygen/structLICM.html) pass uses `AliasSetTracker`s to calculate alias sets for each loop nest. If an `AliasSet` in a loop is not modified, then all load instructions from that set may be hoisted out of the loop. If any alias sets are stored to **and** are must alias sets, then the stores may be sunk to outside of the loop, promoting the memory location to a register for the duration of the loop nest. Both of these transformations only apply if the pointer argument is loop-invariant.

#### The AliasSetTracker implementation

The AliasSetTracker class is implemented to be as efficient as possible. It uses the union-find algorithm to efficiently merge AliasSets when a pointer is inserted into the AliasSetTracker that aliases multiple sets. The primary data structure is a hash table mapping pointers to the AliasSet they are in.

The AliasSetTracker class must maintain a list of all of the LLVM `Value*`s that are in each AliasSet. Since the hash table already has entries for each LLVM `Value*` of interest, the AliasesSets thread the linked list through these hash-table nodes to avoid having to allocate memory unnecessarily, and to make merging alias sets extremely efficient (the linked list merge is constant time).

You shouldn't need to understand these details if you are just a client of the AliasSetTracker, but if you look at the code, hopefully this brief description will help make sense of why things are designed the way they are.

### Using the `AliasAnalysis` interface directly

If neither of these utility class are what your pass needs, you should use the interfaces exposed by the `AliasAnalysis` class directly. Try to use the higher-level methods when possible (e.g., use mod/ref information instead of the [alias](#alias) method directly if possible) to get the best precision and efficiency.

## Existing alias analysis implementations and clients

If you're going to be working with the LLVM alias analysis infrastructure, you should know what clients and implementations of alias analysis are available. In particular, if you are implementing an alias analysis, you should be aware of the [the clients](#the clients) that are useful for monitoring and evaluating different implementations.

### Available `AliasAnalysis` implementations {#various alias analysis implementations}

This section lists the various implementations of the `AliasAnalysis` interface. All of these `chain <aliasanalysis-chaining>`{.interpreted-text role="ref"} to other alias analysis implementations.

#### The `-basic-aa` pass

The `-basic-aa` pass is an aggressive local analysis that *knows* many important facts:

- Distinct globals, stack allocations, and heap allocations can never alias.
- Globals, stack allocations, and heap allocations never alias the null pointer.
- Different fields of a structure do not alias.
- Indexes into arrays with statically differing subscripts cannot alias.
- Many common standard C library functions [never access memory or only read memory](#never access memory or only read memory).
- Pointers that obviously point to constant globals "`pointToConstantMemory`".
- Function calls can not modify or references stack allocations if they never escape from the function that allocates them (a common case for automatic arrays).

#### The `-globalsmodref-aa` pass

This pass implements a simple context-sensitive mod/ref and alias analysis for internal global variables that don't "have their address taken". If a global does not have its address taken, the pass knows that no pointers alias the global. This pass also keeps track of functions that it knows never access memory or never read memory. This allows certain optimizations (e.g. GVN) to eliminate call instructions entirely.

The real power of this pass is that it provides context-sensitive mod/ref information for call instructions. This allows the optimizer to know that calls to a function do not clobber or read the value of the global, allowing loads and stores to be eliminated.

???+note "Note"
    This pass is somewhat limited in its scope (only support non-address taken globals), but is very quick analysis.

#### The `-steens-aa` pass

The `-steens-aa` pass implements a variation on the well-known "Steensgaard's algorithm" for interprocedural alias analysis. Steensgaard's algorithm is a unification-based, flow-insensitive, context-insensitive, and field-insensitive alias analysis that is also very scalable (effectively linear time).

The LLVM `-steens-aa` pass implements a "speculatively field-**sensitive**" version of Steensgaard's algorithm using the Data Structure Analysis framework. This gives it substantially more precision than the standard algorithm while maintaining excellent analysis scalability.

???+note "Note"
    `-steens-aa` is available in the optional "poolalloc" module. It is not part of the LLVM core. :::

#### The `-ds-aa` pass

The `-ds-aa` pass implements the full Data Structure Analysis algorithm. Data Structure Analysis is a modular unification-based, flow-insensitive, context-**sensitive**, and speculatively field-**sensitive** alias analysis that is also quite scalable, usually at `O(n * log(n))`.

This algorithm is capable of responding to a full variety of alias analysis queries, and can provide context-sensitive mod/ref information as well. The only major facility not implemented so far is support for must-alias information.

???+note "Note"
    `-ds-aa` is available in the optional "poolalloc" module. It is not part of the LLVM core. :::

#### The `-scev-aa` pass

The `-scev-aa` pass implements AliasAnalysis queries by translating them into ScalarEvolution queries. This gives it a more complete understanding of `getelementptr` instructions and loop induction variables than other alias analyses have.

### Alias analysis driven transformations

LLVM includes several alias-analysis driven transformations which can be used with any of the implementations above.

#### The `-adce` pass

The `-adce` pass, which implements Aggressive Dead Code Elimination uses the `AliasAnalysis` interface to delete calls to functions that do not have side-effects and are not used.

#### The `-licm` pass

The `-licm` pass implements various Loop Invariant Code Motion related transformations. It uses the `AliasAnalysis` interface for several different transformations:

- It uses mod/ref information to hoist or sink load instructions out of loops if there are no instructions in the loop that modifies the memory loaded.
- It uses mod/ref information to hoist function calls out of loops that do not write to memory and are loop-invariant.
- It uses alias information to promote memory objects that are loaded and stored to in loops to live in a register instead. It can do this if there are no may aliases to the loaded/stored memory location.

#### The `-argpromotion` pass

The `-argpromotion` pass promotes by-reference arguments to be passed in by-value instead. In particular, if pointer arguments are only loaded from it passes in the value loaded instead of the address to the function. This pass uses alias information to make sure that the value loaded from the argument pointer is not modified between the entry of the function and any load of the pointer.

#### The `-gvn`, `-memcpyopt`, and `-dse` passes

These passes use AliasAnalysis information to reason about loads and stores.

### Clients for debugging and evaluation of implementations {#the clients}

These passes are useful for evaluating the various alias analysis implementations. You can use them with commands like:

```bash
opt -ds-aa -aa-eval foo.bc -disable-output -stats
```

#### The `-print-alias-sets` pass

The `-print-alias-sets` pass is exposed as part of the `opt` tool to print out the Alias Sets formed by the [AliasSetTracker](#AliasSetTracker) class. This is useful if you're using the `AliasSetTracker` class. To use it, use something like:

```bash 
opt -ds-aa -print-alias-sets -disable-output
```

#### The `-aa-eval` pass

The `-aa-eval` pass simply iterates through all pairs of pointers in a function and asks an alias analysis whether or not the pointers alias. This gives an indication of the precision of the alias analysis. Statistics are printed indicating the percent of no/may/must aliases found (a more precise algorithm will have a lower number of may aliases).

## Memory Dependence Analysis

???+note "Note"
    We are currently in the process of migrating things from `MemoryDependenceAnalysis` to `MemorySSA`{.interpreted-text role="doc"}. Please try to use that instead.

If you're just looking to be a client of alias analysis information, consider using the Memory Dependence Analysis interface instead. MemDep is a lazy, caching layer on top of alias analysis that is able to answer the question of what preceding memory operations a given instruction depends on, either at an intra- or inter-block level. Because of its laziness and caching policy, using MemDep can be a significant performance win over accessing alias analysis directly.
