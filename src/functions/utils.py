from typing import Callable

import numpy as np

from polars import DataFrame, Series


def map_numpy_column(df: DataFrame, column_name: str, f: Callable[[np.ndarray], np.ndarray], target_columns: str = None) -> DataFrame:

    # map array column
    x = np.array(df[column_name])
    y = f(x)

    # replace source column
    if target_columns is None:
        return df.with_columns(Series(column_name, y))

    # ... or create new columns
    n_columns = y.shape[1]
    result = DataFrame(df)
    for i in range(n_columns):
        result = result.with_columns(Series(f"{target_columns}_{i}", y[:, i]))

    return result
