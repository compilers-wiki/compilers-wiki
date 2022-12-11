# Write a Hello World Pass

In this section, we guide you through writing and running a simple pass that prints the names of all functions contained in the input IR module.

## Setup the Build

Let's assume that you have cloned the [LLVM monorepo](https://github.com/llvm/llvm-project) and setup the build directory at `build/` [^how2build], and the current working directory is the root of the monorepo. In this section, we will build the new pass in-tree, which means that the source tree of the new pass is directly embedded in the source tree of the LLVM monorepo.

[^how2build]: For instructions on how to build LLVM from source, please refer to the [official documentation](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm).

Create a `Hello.h` file under `llvm/include/llvm/Transforms/Utils/`:

```sh
touch llvm/include/llvm/Transforms/Utils/Hello.h
```

Create a `Hello.cpp` file under `llvm/lib/Transforms/Utils/`:

```sh
touch llvm/lib/Transforms/Utils/Hello.cpp
```

Modify `llvm/lib/Transforms/Utils/CMakeLists.txt` and add `Hello.cpp` to the `add_llvm_component_library` call:

```CMake
add_llvm_component_library(LLVMTransformUtils
    # Other files ...
    Hello.cpp
    # Other files and configurations ...
)
```

This step adds `Hello.cpp` to the LLVM build process.

## Code Implementation

Edit `Hello.h` as follows:

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

Edit `Hello.cpp` as follows:

```cpp
#include "llvm/Transforms/Utils/Hello.h"

namespace llvm {

PreservedAnalyses HelloPass::run(Function &F, FunctionAnalysisManager &AM) {
  errs() << F.getName() << "\n";
  return PreservedAnalyses::all();
}

}  // namespace llvm
```

Yes! This is (almost) all you need to write a simple but working pass. Let's break the code into pieces and see how it's working.

To create a pass in C++, you need to write a class that implements the software interface of a pass. Unlike traditional approaches which rely on class inheritance, the new pass manager uses _concepts-based polymorphism_ [^concepts-based-polymorphism1] [^concepts-based-polymorphism2]. **As long as the class contains a `run` method that allows it to run on some piece of IR, it is a pass**. No need to inherit the class from some base class and override some virtual functions. In the code above, our `HelloPass` class inherits from the `PassInfoMixin` class, which adds some boilerplate code to the pass. But the most important part is the `run` method that makes `HelloPass` a pass.

[^concepts-based-polymorphism1]: You can refer to [this comment](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/PassManager.h#L27-L33) for a brief introduction to the concepts-based polymorphism used in the pass framework.
[^concepts-based-polymorphism2]: For a detailed introduction and discussion about concepts-based polymorphism, please refer to [GuillaumeDua. Concept-based polymorphism in modern C++](https://gist.github.com/GuillaumeDua/b0f5e3a40ce49468607dd62f7b7809b1).

The `run` method takes two parameters. The first parameter `F` is the IR function that the pass is running on. Note that `F` is passed via a non-const reference, indicating that we can modify the function (i.e. perform transformations) in the pass. The second parameter `AM` is a pass manager instance that links to analysis passes and provides function-level analysis information.

Since the `run` method takes `Function` as input, `HelloPass` is a **function pass**. The pass manager schedules a function pass to run on every function in the input IR module. When the `HelloPass` gets executed, it writes the function's name to the standard error and finishes.

The `run` method returns a `PreservedAnalyses` object. This object contains information about whether the analysis performed by a previous analysis pass is still valid after this pass runs. The `run` method returns `PreservedAnalyses::all()` to indicate that all available analysis is still valid after running `HelloPass` (because it doesn't modify the IR).

## Register the Pass

We have finished implementing the simple pass but we havn't told LLVM pass manager about the existance of our new pass. We need to _register_ our new pass into the pass manager.

Edit `llvm/lib/Passes/PassRegistry.def` and add the following lines to it:

```cpp
FUNCTION_PASS("hello", HelloPass())
```

Note the first argument to `FUNCTION_PASS` is the name of our new pass.

Add the following `#include` to `llvm/lib/Passes/PassBuilder.cpp`:

```cpp
#include "llvm/Transforms/Utils/Hello.h"
```

and it's done. Now time for building and running our new pass.

## Build and Run

Go to the build directory and build `opt`, which is a dedicated tool for running passes over a piece of IR. After building `opt`, create a test IR file `test.ll` for testing:

```llvm
define i32 @foo() {
  %a = add i32 2, 3
  ret i32 %a
}

define void @bar() {
  ret void
}
```

Then run our new pass with `opt`:

```sh
build/bin/opt -disable-output test.ll -passes=hello
```

The `-passes=hello` option make `opt` run `HelloPass`.

Expected output:

```
foo
bar
```

Congratulations! You have finished your LLVM pass!
