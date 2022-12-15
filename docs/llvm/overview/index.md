# Getting Started

In this section, we will briefly describe the basics and concepts related to compilers and introduce the LLVM project.

???+note "Note"
    This section assumes that you do not have any background in compilers and that you already have an understanding of programming and can write simple C code.

## What is a compiler

Conceptually, a compiler is a **computer program** that converts a programming language into another language. What we usually call a compiler generally converts a higher-level programming language such as C, C++, Fortran, etc. into a lower-level assembly language or machine code.

**Note that a compiler is not an editor, much less an IDE.**

## LLVM

???+note "Note"
    Note that LLVM as discussed here refers to all the subprojects contained in the LLVM monorepo, not to the LLVM core in the traditional sense.

LLVM is a set of **reusable** compiler toolchains. It currently contains at least the following subprojects.

* bolt

A linked optimizer that optimizes the code layout of an application for improvements based on execution profiles collected by sampled profilers (such as the Linux perf tool).

* clang

A GCC-compatible compiler front-end for the C family (C, C++, Objective C/C++, OpenCL, CUDA), known for its respectable compilation times and good diagnostic information.

* libcxx

A new implementation of the C++ standard library, for C++11 and above.


* llvm

The core of the LLVM project, including the LLVM intermediate language, various optimizations, and its corresponding command-line tools.

* mlir

A general framework for writing various types of analysis and transformations in compilers and the intermediate representations they require.

* lld

A high-performance, modular cross-platform linker.

## LLVM design philosophy

LLVM is modular, and the most important aspect of LLVM is that it is designed as a set of libraries, not as a single command-line compiler like GCC or an opaque virtual machine like the JVM or .NET virtual machines.

As an example, the LLVM optimizer provides a large number of [passes](../pass/index.md) that are compiled into one or more `.o` files and then built into a series of static or dynamic libraries. These libraries provide various parsing and transformation functions, and the passes are kept as independent as possible from each other, or if they depend on other parsing to do their job, their dependencies on other passes are explicitly declared. In contrast, during the actual optimization process of LLVM optimization, only the passes that are used are linked to the final application, not all of them.

This straightforward design approach allows LLVM to provide a large number of features, some of which may be useful only to a specific audience, without affecting users of the library who just want to do other simple things. In contrast, traditional compiler optimizers are built as large amounts of interrelated code, making it difficult to split, reason about, and accelerate them. With LLVM, you can understand the individual optimizers without knowing how the whole system fits together.

## Pipeline of a compiler (taking Clang/LLVM as an example)

Generally speaking, compilers are divided into front-end, middle-end and back-end.

The front-end is responsible for analyzing and processing the source code and generating intermediate code.

### Lexing

In this stage, the compiler divides the source code into tokens and ignores invalid information such as spaces and blank lines. Each token usually contains the following information.

* Type. Common types such as identifiers, keywords, separators, operators, literals, etc.
* Value. The value that this token actually corresponds to.
* Location in the source code. Recording the location of the token in the original code can help us generate better diagnostic information.

As an example:
```c
int foo(int i) {
    return i + 42;
}

int main() {
    int x = foo(24);
    return x - 24;
}
```
For the above code, Clang will generate the following token:

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

### Parsing

In this stage, the compiler converts the linear token stream into an Abstract Syntax Tree with specific syntax rules. The compiler also throws a diagnostic at this stage if there is an error in the syntax parsing.

For the preceding code, the syntax tree generated by Clang can be viewed with the following command.
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
### Intermediate code generation

At this stage, the compiler converts the abstract syntax tree into an intermediate language.

As in the example above, Clang generates the following intermediate language (LLVM IR).
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

The middle-end is mainly responsible for analyzing and optimizing on the intermediate code to generate a more efficient intermediate code. In addition to the generic optimizations, the middle-end also contains those platform-specific optimizations.

The specific responsibilities of the middle-end include:

* Analysis of the intermediate language, such as data flow analysis, alias analysis, etc. This phase is the basis for other optimizations performed by the compiler.
* Optimizing the intermediate language to convert its representation into a functionally equivalent but faster (or smaller) form. Common optimizations such as inline expansion, dead code elimination, constant propagation, loop conversion, etc.

For example, we have this LLVM IR:
```llvm
$ cat prog.ll
define void @test(ptr %Q, ptr %P) {
  %DEAD = load i32, ptr %Q
  store i32 %DEAD, ptr %P
  store i32 0, ptr %P
  ret void
}
```

We can use the command line tools provided by LLVM to optimize it in any way we want. By working with the pipeline character on Unix systems, we can be very flexible in achieving various effects:
```sh
$ opt -passes=dse prog.ll | llvm-dis -f
; ModuleID = '<stdin>'
source_filename = "prog.ll"

define void @test(ptr %Q, ptr %P) {
  store i32 0, ptr %P, align 4
  ret void
}
```

Here we use the `opt` utility provided by LLVM to apply a dead code elimination optimization to `prog.ll` alone. Since `opt` generates human-unreadable LLVM bytecode, we then use `llvm-dis` to convert it to LLVM IR.

The backend needs to generate the intermediate code into machine code for the corresponding platform, as well as some CPU architecture details for optimization.

## How to contribute to LLVM

### Submitting new bugs

For a long time LLVM has been using [Bugzilla](https://bugs.llvm.org/) as its bug tracker, but has now moved to [Github issues](https://github.com/llvm/llvm-project/issues). You can find some easy-to-follow bugs by looking at [Good Frist Issue](https://github.com/llvm/llvm-project/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) in issue.

### Submit a patch

???+note "Note"
    Note that LLVM is interested in dropping Phabricator and moving to Github Pull Requests, as detailed in [^Code Review Process Update].

[^Code Review Process Update]: You can refer to [this post](https://discourse.llvm.org/t/code-review-process-update/63964) to know more.

Currently, the LLVM community still uses [Phabricator](https://reviews.llvm.org/) to submit patches for code review. You can quickly sign up for a Phabricator account using a third-party account such as GitHub.

You can submit your patches using the web page or Phabricator's own command line tool, arcanist. Specific documentation can be found [here](https://llvm.org/docs/Phabricator.html#phabricator-reviews). The following is a simple example of using arcanist.

???+note "Note"
    Here we assume that you are using a Unix-like operating system and have Git installed and configured

First we need to install arcanist:
```sh
git clone https://github.com/phacility/arcanist.git
```

Next, add it to the environment variables:
```sh
export PATH="$PATH:/path/to/arcanist/bin/"
```

Authorize the login.
```sh
arc install-certificate
```

When coding, we recommend using ``amend`` or manually ``rebase`` to compress our code into a ``commit`` for subsequent code review.

Once our patch has been tested locally, it's ready to be sent out. But before doing so, we must be careful to ensure that our patch conforms to the LLVM Coding Style with clang-format.

We recommend using the official [git-clang-format](https://github.com/llvm/llvm-project/blob/main/clang/tools/clang-format/git-clang-format) tool provided by LLVM for best results and to avoid accidental changes to other code. You can add the path of git-clang-format to the PATH directly.

```sh
git clang-format HEAD~1
```

Note that the format doesn't commit automatically, so you need to.
```sh
git commit --amend -a
```

After this is done, we can commit the patch to Phabricator using arc
```sh
arc diff
```
arcanist will upload the latest commit to Phabricator, and when it succeeds, it will give us a link to our Revision.

After the reviewer has looked at the code, he may give us some suggestions for changes, and we can continue to use arcanist to update our patches:
```sh
arc diff --update DXXXXX
```
where DXXXXX is our previous Revision, which we have to specify manually because arcanist doesn't know the context.

### Communicating with the community

* Mailing list

Although the community has phased out mailing lists, we still recommend that you subscribe to the list of commit records for the corresponding subprojects, which helps to keep you informed of what is happening in the project. For example, if you want to contribute to Clang, you can subscribe to the list [cfe-commits](https://lists.llvm.org/mailman/listinfo/cfe-commits), and if you want to contribute to the LLVM core, you can subscribe to the list [llvm-commits](https://lists.llvm.org/mailman/listinfo/llvm-commits).

* [Discourse](https://discourse.llvm.org/)
* [Discord](https://discord.com/invite/xS7Z362)

### Related links

* [Getting Started with the LLVM System](https://llvm.org/docs/GettingStarted.html)
* [The Design Decisions of LLVM](https://www.aosabook.org/en/llvm.html)
* [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html)
* [LLVM IR Tutorial - Phis, GEPs and other things, oh my!](https://youtu.be/m8G_S5LwlTo)
* [2019 LLVM Developers’ Meeting: E. Christopher & J. Doerfert “Introduction to LLVM”](https://youtu.be/J5xExRGaIIY)
* [“Clang” CFE Internals Manual](https://clang.llvm.org/docs/InternalsManual.html)
* [2019 LLVM Developers’ Meeting: S. Haastregt & A. Stulova “An overview of Clang ”](https://youtu.be/5kkMpJpIGYU)
* [P. Goldsborough “clang-useful: Building useful tools with LLVM and clang for fun and profit"](https://youtu.be/E6i8jmiy8MY)