import matplotlib.pyplot as plt
import numpy as np
from polars import Series

from sklearn.decomposition import PCA

from dho_analysis.calc_message_embeddings import add_message_embeddings
from dho_analysis.calc_sentences import explode_msg_to_sentences
from dho_analysis.calc_slow_features import add_sfa_columns
from dho_analysis.load_practice_logs_for_author import load_practice_logs_for_author
from dho_analysis.load_time_aggregated_practice_logs import load_time_aggregated_practice_logs, \
    aggregate_messages_by_time
from dho_analysis.utils import SEED


def main():
    plot_slowness()


def plot_slowness(
        author: str = "Linda ”Polly Ester” Ö",
        model: str = "all-MiniLM-L6-v2",
        time_aggregate: str = "1w",
        pca_components: int = 20,
):

    df0 = load_practice_logs_for_author(author=author)

    df_sen = explode_msg_to_sentences(df=df0)
    df_sen = add_message_embeddings(df=df_sen, model=model)

    pca = PCA(n_components=pca_components, random_state=SEED)
    pca.fit(np.array(df_sen["embedding"]))

    df_agg = aggregate_messages_by_time(df=df0, time_aggregate=time_aggregate)
    df_agg = add_message_embeddings(df=df_agg, model=model)

    old_embeddings: Series = df_agg["embedding"]
    new_embeddings: Series = Series("embedding", pca.transform(np.array(old_embeddings)))
    df_agg = df_agg.with_columns(new_embeddings)

    df_agg = add_sfa_columns(df_agg)
    plt.plot(df_agg.select(["date"]), df_agg.select(["SFA_0"]))
    plt.show()


if __name__ == "__main__":
    main()
