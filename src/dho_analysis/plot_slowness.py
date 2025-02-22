import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from polars import Series, DataFrame
from sklearn.decomposition import PCA
from sksfa import SFA

from dho_analysis.calc_message_embeddings import add_message_embeddings
from dho_analysis.load_time_aggregated_practice_logs import load_time_aggregated_practice_logs


def main():
    plot_slowness()


def plot_slowness():
    df = load_time_aggregated_practice_logs(time_aggregate="1w")
    df = add_message_embeddings(df)
    df = add_sfa_columns(df)
    plt.plot(df.select(["date"]), df.select(["SFA_0"]))
    plt.show()


def add_sfa_columns(df: DataFrame) -> DataFrame:
    df_sfa = _calc_sfa(df.get_column("embedding"))
    return pl.concat([df, df_sfa], how="horizontal")


def _calc_sfa(x: Series, pca=.98) -> DataFrame:
    components = PCA(n_components=pca).fit_transform(np.array(x))
    y = SFA(n_components=3).fit_transform(components)[:,:3]
    return DataFrame(y, schema=["SFA_0", "SFA_1", "SFA_2"])



if __name__ == "__main__":
    main()
