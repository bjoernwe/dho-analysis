from typing import Callable

import numpy as np

from polars import DataFrame, Series


def map_numpy_column(df: DataFrame, column_name: str, f: Callable[[np.ndarray], np.ndarray]) -> DataFrame:
    return df.with_columns(Series(column_name, f(np.array(df[column_name]))))
