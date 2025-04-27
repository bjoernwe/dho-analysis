from typing import Optional

import numpy as np
import polars as pl

from polars import Series, DataFrame
from sklearn.decomposition import PCA
from sksfa import SFA

from data.load_time_aggregated_practice_logs import load_time_aggregated_practice_logs
from config import SEED
from models.SentenceTransformerModel import SentenceTransformerModel


def main():
    print_example_slow_features()


def print_example_slow_features():
    model = SentenceTransformerModel("all-MiniLM-L6-v2")
    df = load_time_aggregated_practice_logs(
        author="Linda ”Polly Ester” Ö",
        model=model,
        time_aggregate_every="1d",
        time_aggregate_period="1w",
    )
    df = add_sfa_columns(df, pca=.98)
    print(df)


def add_sfa_columns(df: DataFrame, pca: Optional[float] = None) -> DataFrame:
    df_sfa = _calc_sfa(df.get_column("embedding"), pca=pca)
    return pl.concat([df, df_sfa], how="horizontal")


def _calc_sfa(x: Series, pca: float) -> DataFrame:
    data = np.array(x)
    if pca is not None and 0 < pca < 1:
        data = PCA(n_components=pca, random_state=SEED).fit_transform(data)
    y = SFA(n_components=3).fit_transform(data)[:,:3]
    return DataFrame(y, schema=["SFA_0", "SFA_1", "SFA_2"])



if __name__ == "__main__":
    main()
