import matplotlib.pyplot as plt
import numpy as np
from polars import Series

from sklearn.decomposition import PCA
from sksfa import SFA

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
        model: str = "all-mpnet-base-v2",
        time_aggregate: str = "1d",
        pca_components: int = 50,
):

    df0 = load_practice_logs_for_author(author=author)
    df0 = df0.sort("date")

    df_sen = explode_msg_to_sentences(df=df0)
    df_sen = add_message_embeddings(df=df_sen, model=model)

    df_agg = aggregate_messages_by_time(df=df0, time_aggregate=time_aggregate)
    df_agg = add_message_embeddings(df=df_agg, model=model)

    pca = PCA(n_components=pca_components, random_state=SEED)
    pca.fit(np.array(df_sen["embedding"]))

    df_sen = df_sen.with_columns(Series("embedding", pca.transform(np.array(df_sen["embedding"]))))
    df_agg = df_agg.with_columns(Series("embedding", pca.transform(np.array(df_agg["embedding"]))))

    sfa = SFA(n_components=1, random_state=SEED)
    sfa.fit(np.array(df_agg["embedding"]))

    df_sen = df_sen.with_columns(Series("SFA", sfa.transform(np.array(df_sen["embedding"]))))
    df_agg = df_agg.with_columns(Series("SFA", sfa.transform(np.array(df_agg["embedding"]))))

    for s in df_sen.sort("SFA_0")["msg"].to_list()[:10]: print(s)
    print("\n...\n")
    for s in df_sen.sort("SFA_0")["msg"].to_list()[-10:]: print(s)

    plt.plot(df_agg.select(["date"]), df_agg.select(["SFA"]))
    plt.show()


if __name__ == "__main__":
    main()
