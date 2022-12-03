# Multi-Level Intermediate Representation

**The MLIR project is developing rapidly, and the situation may change at any time, please pay attention to the writing time of the page when reading**.

The MLIR project is a novel approach to building reusable and extensible compiler infrastructure. MLIR aims to address software fragmentation, improve compilation for heterogeneous hardware, significantly reduce the cost of building domain-specific compilers, and aid in connecting existing compilers together.

In simple terms, MLIR is a general framework for writing various types of analyzes and transforms in compilers and the intermediate representations they require. It provides a series of facilities for compiler developers to define new IRs and use existing IRs conveniently and quickly. Developers can easily mix various IRs, freely converting between them, and make different levels of analysis to finally generate the object code.

The MLIR project also contains some pre-written "IRs" ready for immediate use. You can start to understand these existing "IRs" and learn how to build your own "IR" from the basic concepts and building blocks of MLIR: [dialect](basic/dialect.md).

## Introduction to this section of the wiki

This section of the wiki aims to:

- Aids in the development of MLIR itself
- Assist in the development of new dialects based on MLIR
- Assist in the use of MLIR's existing dialects and passes

## Related Links

- [Official Documentation](https://mlir.llvm.org/docs/)
- [CodeGen Dialect Overview](https://discourse.llvm.org/t/codegen-dialect-overview/2723)
    - A post introducing the descent path of each dialect in MLIR. Although it is a bit old and some dialects have been dismantled, the classification is worth pondering. **A picture is worth a thousand words**, and you should definitely read the pictures even if you are too busy reading the words.
    - The author classifies MLIR with two dimensions: data and algorithm. From Tensor -> Buffer, the data gradually changes from abstract tensor to memory. From Structure -> Payload, algorithms from high-level, structured descriptions to low-level, imperative codes
    - The author also shared the concerns of each dialect
