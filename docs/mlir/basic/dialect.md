# Dialect

Dialects are a crucial part of MLIR. It can be said that MLIR without various dialects is like a language without a standard library, and will not have any practical use. If the MLIR framework itself is compared to the foundation, then the various dialects built on top of MLIR are buildings with different styles. Data and information flow within and between buildings, forming a city with ducks in a row.

As far as learning is concerned, the learning of dialects can be divided into two parts:

- Learn how to construct your own dialect
- Learn the meaning of various operations in the built-in dialect

The former is like learning to build a building, and the latter is like visiting a "model house".

## Basic Concepts

A dialect can basically completely define anything that belongs to it, like new operations, properties, and types. New operations in the dialect can define their own exclusive syntax format and output format. It is very reasonable and normal to see operations with unique text forms in the operations of different dialects.

Things from different dialects can coexist harmoniously in the same module/function. At the same time, dialect writers can provide passes that convert operations in other dialects to their own dialects, and can also write passes that convert operations in their own dialects to other dialects. These conversions can be complete, converting all operations in A dialect transfer all operations to B; or step by step, only transfer part of the operations in A to B, and then transfer the rest to C.

Consider the following example:

```mlir
// Define a function using the func operator in the func dialect
// Using a custom syntax, the textual form of the operation looks similar to a function definition in a normal language
func. func @main() {
  // use constant operations in the arith dialect
  // using the basic syntax
  %lb = arith.constant 0 : index
  %ub = arith. constant 5 : index

  %step = arith. constant 1 : index

  %sum_0 = arith.constant 0.0 : f32
  %t = arith.constant 5.0 : f32

  // use the for operator in the scf dialect
  // It has custom syntax and can be written to look like "trailing closures" in other languages
  // takes a "function" as its own parameter
  %sum = scf.for %iv = %lb to %ub step %step
    iter_args(%sum_iter = %sum_0) -> (f32) {
        %1 = arith.addf %sum_iter , %t : f32
        scf. yield %1 : f32
  }

  // use the print operation in the vector dialect
  vector.print %sum : f32
  return
}
```

After this example is written, various conversion passes can be used to gradually or directly convert operations in various dialects into operations in LLVM dialects, and then translate into LLVM IR to obtain executable files. It can be seen that each dialect is very free. So to read MLIR code text, it is not enough to only be familiar with the MLIR core framework. You must refer to the documentation of the dialect you want to use, and be familiar with the meaning and syntax of each operation in them, in order to read efficiently.

All dialects are equal, but some dialects are more equal, and they are the dialects that are shipped in the MLIR project repository. These dialects are widely used. Before learning other dialects, it is recommended to read the documentation of [some of the most commonly used built-in dialects] (#Common Dialects).

## Build a New Dialect

This part can refer to the official [Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/).

## Learn the Built-in Dialect

Currently the MLIR official repository contains some built-in dialects and conversions between them. These dialects can be regarded as MLIR's "standard library". Depending on the degree of versatility and focus, they can be classified as follows:

> Due to the rapid development of MLIR, the list of dialects below will not be guaranteed to be complete. For a complete list, please refer to the official documentation.

### Common Dialects

The following are very common dialects. Basically, any other built-in dialects will more or less use the operations in the above dialects. Learning them is an important part of understanding existing dialects:

- `builtin`: built-in dialect, contains some facilities that may be used in all dialects, and should theoretically be kept very small. It also contains type definitions for the basic types of each built-in dialect
- `func`: function dialect, used to express function definition/function call
- `arith`: arithmetic dialect, including various basic arithmetic operations
- `affine`: affine transformation dialect, which is a high-level description of index space/address space mapping
- `memref`: memory area dialect, which is a high-level abstraction of memory and various operations on it

There are also some less common but common dialects:

- `cf`: Express unstructured control flow, such as goto and the like
- `scf`: express structured control flow, such as for/while/if
- `math`: common numerical operation functions
- `complex`: commonly used complex numerical operation functions
- `index`: platform-independent operations on indexed types, perhaps comparable to "high-level pointer arithmetic"

### LLVM Dialects

Then there is a very specific dialect: `llvm`, the LLVM dialect. It's basically a dialect that maintains a one-to-one mapping with concepts in LLVM IR. It acts as the "lowest level" for most dialect descents Dialect, other dialects gradually descend to the LLVM dialect, and then translate the LLVM dialect into LLVM IR, and then have various optimizations of LLVM and the ability to translate to all backends of LLVM.

If you don't understand the operation in any dialect, and it also provides a lowering pass, you can read the lowering pass to understand what it means. The whole process is similar to learning C language from a C language compiler. In this process, the LLVM dialect acts like an assembly language.

### Domain Specific Dialects

Then there are some domain-specific dialects, most of which appear to effectively express information highly related to a certain domain at an abstract high level, so an understanding of specific domains will play a great role in understanding the following built-in dialects help. They are:

- `vector`: a vectorized dialect, designed to express information about various SIMD operations in a platform-independent manner
- `linalg`: high-level abstraction related to deep learning model representation
- `spirv`: support for [SPIR-V](https://registry.khronos.org/SPIR-V/), a dialect for expressing graphics domain concepts such as graphics shaders
- `omp`: support for [OpenMP](https://www.openmp.org/) for expressing automatic multithreading related concepts
- `gpu`: GPU dialect designed to express information about operations on the GPU platform-independently
- `async`: dialect for expressing asynchronous operations
- `pdl`, `pdl_interp`: MLIR dialect used to express MLIR transformation, convenient to use MLIR pass to analyze MLIR pass :D

### Platform-dependent Dialects

Among the built-in dialects, there are also many dialects for underlying optimization. They are designed to describe the characteristics of a certain CPU or instruction set in detail and guide the underlying optimization. Their names are relatively intuitive:

- GPU:
    - `amdgpu`
    - `nvgpu`
    - `nvvm`: NVVM IR for CUDA
- ARM:
    - `arm_neon`
    - `arm_sve`
- Intel:
    - `amx`: The Intel Advanced Matrix Extensions
    - `x86vector`
