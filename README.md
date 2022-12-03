# LLVM Wiki

## Why?

Compiler optimizations and related principles are taught in university courses and university textbooks, while there are a large number of documents for program optimization performed by compilation technology, but they are scattered everywhere and cannot be studied intensively. This warehouse is a description document that contains a large number of compiler-related optimization technologies. The content is mainly high-level design principles and design implementations.

## Installation and testing

Our project is built using [Mkdocs](https://www.mkdocs.org/).

Firstly clone our repo, you may need a shallow clone via `--depth=1`


```sh
git clone --depth=1 https://github.com/llvm-wiki/llvm-wiki.git
```

```sh
cd llvm-wiki
```

Install dependencies

```sh
pipenv install
```


To preview the edited version, run

```sh
pipenv run mkdocs serve -v
```

then visit `http://localhost:8000` .

## Contribute to this repository

We currently accept Github Pull Requests.

## License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
