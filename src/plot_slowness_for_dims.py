import random

import matplotlib.pyplot as plt
import numpy as np

from polars import Series

from data.load_practice_logs_for_author import load_practice_logs_for_author
from data.load_time_aggregated_practice_logs import aggregate_messages_by_time
from functions.calc_message_embeddings import add_message_embeddings, add_mock_message_embeddings
from functions.calc_sentences import explode_msg_to_sentences
from functions.calc_slowness_value import calc_slowness_for_array, calc_slowness_for_series
from models.ZeroShotEmbeddingTransformer import ZeroShotEmbeddingTransformer
from plot_slowness import zeroshot_labels


def main():
    plot_slowness_depending_on_dims()


def plot_slowness_depending_on_dims(
        author: str = "Linda ”Polly Ester” Ö",
        time_aggregate: str = "1d",
):

    # Load practice logs (per sentence)
    df0 = load_practice_logs_for_author(author=author)
    df0 = df0.sort("date")
    df_sen = explode_msg_to_sentences(df=df0)

    # Calc embedding for each sentence in logs
    labels = random.sample(zeroshot_labels, len(zeroshot_labels))
    model = ZeroShotEmbeddingTransformer(model="facebook/bart-large-mnli", labels=labels, batch_size=1000)
    df_embeddings = add_message_embeddings(df=df_sen, model=model)
    df_mock_embeddings = add_mock_message_embeddings(df=df_sen, dims=len(labels))

    # Aggregate messages and embeddings time-wise
    df_agg = aggregate_messages_by_time(df=df_embeddings.select(["date", "msg", "embedding"]), time_aggregate=time_aggregate)
    df_agg_mock = aggregate_messages_by_time(df=df_mock_embeddings.select(["date", "msg", "embedding"]), time_aggregate=time_aggregate)

    # Plot delta values
    print(labels)
    num_dims = np.arange(1, len(zeroshot_labels))
    deltas = [_calc_slowness(s=df_agg.get_column("embedding"), dims=n) for n in num_dims]
    deltas_mock = [_calc_slowness(s=df_agg_mock.get_column("embedding"), dims=n) for n in num_dims]
    plt.plot(num_dims, deltas)
    plt.plot(num_dims, deltas_mock)
    plt.show()


def _calc_slowness(s: Series, dims: int) -> float:
    a = np.array(s)[:,:dims]
    return calc_slowness_for_array(a)


if __name__ == "__main__":
    main()
