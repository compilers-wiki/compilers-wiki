# Multi-Level Intermediate Representation

**The MLIR project is developing rapidly, and the situation may change at any time, please pay attention to the writing time of the page when reading**.

The MLIR project is a fresh attempt at building a reusable and extensible compiler infrastructure. It aims to solve the problem of software fragmentation, improve compilation on heterogeneous hardware, reduce the development cost of domain-specific compilers, and connect existing compiler facilities.

In simple terms, MLIR is a general framework for writing various types of analyzes in compilers and the intermediate representations they require. It provides a series of facilities for compiler developers to define new IRs and use existing IRs conveniently and quickly. Developers can easily mix and use various IRs, freely switch between them, and make different The level of analysis finally generates the object code.

The MLIR project also contains some pre-written "IRs" ready for immediate use. You can start to understand these existing "IRs" and learn how to build your own "IR" from the basic concepts and building blocks of MLIR: [dialect](basic/dialect.zh.md).

## Introduction to this Wiki

This wiki aims to:

- Aids in the development of MLIR itself
- Assist in the development of new dialects based on MLIR
- Assist in the use of MLIR's own dialect and pass

## Related Links

- [Official Documentation](https://mlir.llvm.org/docs/)
- [CodeGen Dialect Overview](https://discourse.llvm.org/t/codegen-dialect-overview/2723)
    - A post introducing the descent path of each dialect in MLIR. Although it is a bit old and some dialects have been dismantled, the classification is worth pondering. The pictures in it are even more **A picture is worth a thousand words, and you have to read the pictures without reading the words. **
    - The author classifies MLIR according to the two dimensions of data-algorithm. From Tensor -> Buffer, the data gradually changes from abstract tensor to memory. Algorithms go from Structure -> Payload, that is, from high-level, structured descriptions to low-level, imperative things
    - The author also shared the concerns of each dialect
- [How to evaluate the design idea of ​​Linalg Dialect in MLIR project? - mackler's answer - Zhihu](https://www.zhihu.com/question/442964082/answer/1718438248)
    - Comparative "sharp" opinion on MLIR
    - Its classification of compiler optimizations is impressive: auto-tiling/auto-tensorize/auto-schedule
- Some Chinese users who follow MLIR, you can follow the timeline to see
    - [hunterzju](https://www.zhihu.com/people/hunterzju-45): main concern
    - [mackler](https://www.zhihu.com/people/mackler): Mainly biased towards MLIR's work in the AI ​​field, although ta may focus more on generalized computer architecture and optimization
    - [Fasthouse](https://www.zhihu.com/people/zhang-hong-bin-99): The "host" of the Buddy-mlir project, basically the homepage is all about MLIR information
    - [MLIR China Community](https://www.zhihu.com/people/mlir-70): As the name suggests
