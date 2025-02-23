import numpy as np
import polars as pl
from polars import DataFrame

from sklearn.decomposition import PCA

from dho_analysis.calc_message_embeddings import add_message_embeddings
from dho_analysis.calc_sentences import explode_msg_to_sentences
from dho_analysis.load_practice_logs_for_author import load_practice_logs_for_author
from dho_analysis.utils import SEED


def main():
    print_pca_example_sentences(component=0)


def print_pca_example_sentences(component: int = 0):

    df = load_practice_logs_for_author(author="Linda ”Polly Ester” Ö")
    df = explode_msg_to_sentences(df=df)
    df = add_message_embeddings(df=df, model="all-MiniLM-L6-v2")
    df = add_pca_columns(df=df, n_components=component+1)

    col = f"PCA_{component}"
    sentences = list(df.select(["msg", col]).sort(col, descending=True)["msg"])

    for s in sentences[:20]:
        print(s)
    print(" ... ")
    for s in sentences[-20:][::-1]:
        print(s)


def add_pca_columns(df: DataFrame, n_components=10) -> DataFrame:
    pca = PCA(n_components=n_components, random_state=SEED)
    components = pca.fit_transform(np.array(df["embedding"]))
    df_pca = DataFrame(components, schema=[f"PCA_{i}" for i in range(n_components)])
    return pl.concat([df, df_pca], how="horizontal")


if __name__ == "__main__":
    main()
