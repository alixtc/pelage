

![release](https://img.shields.io/github/v/release/alixtc/pelage?color=orange.png)
![coverage](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Falixtc%2Fpelage%2Fmaster%2F.coverage%2Fcoverage.json&query=%24.totals.percent_covered_display&suffix=%25&label=Coverage&color=green)
![Licence](https://img.shields.io/github/license/alixtc/pelage.png)
![python-version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Falixtc%2Fpelage%2Frefs%2Fheads%2Fmaster%2Fpyproject.toml)
![polars-version](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Falixtc%2Fpelage%2Frefs%2Fheads%2Fmaster%2Fpyproject.toml&query=%24.project.dependencies&label=Dependency)
![PyPI -
Downloads](https://img.shields.io/pypi/dm/pelage?color=orange.png)

# Welcome to pelage!

The goal of this project is to provide a simple way to test your
`polars` code on the fly, while doing your analysis. The main idea is to
chain a series of meaningful checks on your data so that you can
continue and be more confident about your data quality. Here is how to
use it:

``` python
import polars as pl

import pelage as plg

data = pl.DataFrame(
    {
        "a": [1, 2, 3],
        "b": ["a", "b", "c"],
    }
)
validated_data = (
    data.pipe(plg.has_shape, (3, 2))
    .pipe(plg.has_no_nulls)
    .with_columns(
        pl.col("a").cast(str).alias("new_a"),
    )
)

print(validated_data)
```

    shape: (3, 3)
    ┌─────┬─────┬───────┐
    │ a   ┆ b   ┆ new_a │
    │ --- ┆ --- ┆ ---   │
    │ i64 ┆ str ┆ str   │
    ╞═════╪═════╪═══════╡
    │ 1   ┆ a   ┆ 1     │
    │ 2   ┆ b   ┆ 2     │
    │ 3   ┆ c   ┆ 3     │
    └─────┴─────┴───────┘

Here is a example of the error messages that if the checks fail:

``` python
try:
    validated_data.pipe(plg.not_accepted_values, {"new_a": ["3"]})
except plg.PolarsAssertError as err:
    print(err)
```

    Details
    shape: (1, 1)
    ┌───────┐
    │ new_a │
    │ ---   │
    │ str   │
    ╞═══════╡
    │ 3     │
    └───────┘
    Error with the DataFrame passed to the check function:
    --> This DataFrame contains values marked as forbidden

Here are the main keys points:

- Each `pelage` check returns the original `polars` DataFrame if the
  data is valid. It allows you continue your analysis by chaining
  additional transformations.

- `pelage` raises an meaningful error message each time the data does
  not meet your expectations.

# Installation

Install the package directly via PIP:

``` bash
pip install pelage
```

# Main Concepts

## Defensive analysis:

The main idea of `pelage` is to leverage your possibility for defensive
analysis, similarly to other python packages such as “bulwark” or
“engarde”. However `pelage` rely mainly on possibility to directly pipe
and chain transformations provided by the fantastic `polars` API rather
than using decorators.

Additionally, some efforts have been put to have type hints for the
provided functions in order to ensure full compatibility with your IDE
across your chaining.

## Leveraging `polars` blazing speed:

Although it is written in python most of `pelage` checks are written in
a way that enable the polars API to work its magic. We try to use a
syntax that is compatible with fast execution and parallelism provided
by polars.

![Site-Readme](assets/presentation.gif)
![Github-Readme](docs/assets/presentation.gif)

## Interoperability:

The polars DSL and syntax have been develop with the idea to make the
transition to SQL much easier. In this perspective, `pelage` wants to
facilitate the use of tests to ensure data quality while enabling a
possible transition towards SQL, and using the same tests in SQL. This
is why we implemented most of the checks that have been developed for
`dbt` tool box, notably:

- [dbt generic
  checks](https://docs.getdbt.com/docs/build/data-tests#generic-data-tests)
- [dbt-utils
  tests](https://github.com/dbt-labs/dbt-utils?tab=readme-ov-file)
- (Soon to come: dbt expectations)

We believe that data quality checks should be written as close as
possible to the data exploration phase, and we hope that providing
theses checks in a context where it is easier to visualize your data
will be helpful. Similarly, we know that it is sometimes much easier to
industrialize SQL data pipelines, in this perspective the similarity
between `pelage` and `dbt` testing capabilities should make the
transition much smoother.

# Why pelage?

`pelage` is the french word designating an animal fur, and particularly
in the case of polar bears, it shields them from water, temperature
variations and act as a strong camouflage. With the skin it constitutes
a strong barrier against the changes in the outside world, and it is
therefore well-suited name for a package designed to help with defensive
analysis.
