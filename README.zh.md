# LLVM Wiki

## 为什么要有这些文档

编译优化和相关原理在大学课程、大学教材中讲授，而利用编译技术执行的程序优化有大量文档，但散落在各处，无法集中学习。这个仓库是包含了大量编译器相关的优化技术的描述文档，内容主要是高层次的设计原理、设计实现。

## 本地安装和测试

我们的仓库使用 [Mkdocs](https://www.mkdocs.org/) 构建。

首先 clone 这个仓库，你可能需要一个 shallow clone 来加快 clone 的速度，传入 `--depth=1`。

```sh
git clone --depth=1 https://github.com/llvm-wiki/llvm-wiki.git
```

```sh
cd llvm-wiki
```

安装相关依赖

```sh
pipenv install
```

运行

```sh
pipenv run mkdocs serve -v
```

然后访问 `http://localhost:8000` 来预览编辑后的版本。

## 贡献本仓库

我们目前接受 Github Pull Requests.

## License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />这项工作的许可证为 <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
