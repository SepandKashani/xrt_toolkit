# Benchmarks

| folder | what it measures | results |
|---|---|---|
| `speed/` | time of projection and back-projection, XTK and other libraries | [speed.md](speed/speed.md) |

Each folder has one script that prints the tables of its page:

    python benchmarks/speed/speed.py

The cases (geometries, sizes) are defined once in `common.py`. Each library has a small
wrapper in `libs/` and runs in its own environment, given by `ASTRA_PYTHON`, `TIGRE_PYTHON`
and `PARALLELPROJ_PYTHON` (default: the current interpreter), for example

    TIGRE_PYTHON="/path/to/tigre-env/bin/python -s" python benchmarks/speed/speed.py

A library that does not run in its environment is shown as "-".

To add a benchmark: a new folder with its script and its page, using the cases of `common.py`.
