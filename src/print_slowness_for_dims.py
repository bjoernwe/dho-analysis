from typing import List, Tuple

import numpy as np
import polars as pl

from polars import Series, DataFrame
from tqdm import tqdm

from config import SEED
from data.load_practice_logs_for_author import load_practice_logs_for_author
from data.load_time_aggregated_practice_logs import aggregate_messages_by_time
from functions.calc_message_embeddings import add_message_embeddings
from functions.calc_sentences import explode_msg_to_sentences
from functions.calc_slowness_value import calc_slowness_for_array
from models.ZeroShotEmbeddingTransformer import ZeroShotEmbeddingTransformer
from plot_slowness import zeroshot_labels


generator = np.random.default_rng(SEED)


def main():
    print_slowness_gain_depending_on_dims()


def print_slowness_gain_depending_on_dims(
        author: str = "Linda ”Polly Ester” Ö",
        time_aggregate: str = "1d",
):

    # Load practice logs (per sentence)
    df0 = load_practice_logs_for_author(author=author)
    df0 = df0.sort("date")
    df_sen = explode_msg_to_sentences(df=df0)

    # Calc embedding for each sentence in logs
    model = ZeroShotEmbeddingTransformer(model="knowledgator/comprehend_it-base", labels=zeroshot_labels, batch_size=1000)
    df_embeddings = add_message_embeddings(df=df_sen, model=model)
    #df_mock_embeddings = add_mock_message_embeddings(df=df_sen, dims=len(zeroshot_labels))

    # Aggregate messages and embeddings time-wise
    df_agg = aggregate_messages_by_time(df=df_embeddings.select(["date", "msg", "embedding"]), time_aggregate=time_aggregate)
    #df_agg_mock = aggregate_messages_by_time(df=df_mock_embeddings.select(["date", "msg", "embedding"]), time_aggregate=time_aggregate)

    # Calc slowness gains for embeddings
    df_gains = _calc_slowness_gains_for_embeddings(embeddings=df_agg["embedding"], labels=zeroshot_labels, n_iterations=100)
    for i in range(len(df_gains)):
        print(f"{df_gains['label'][i]}: {df_gains['mean_delta_diff'][i]}")

    #print(labels)
    #num_dims = np.arange(1, len(zeroshot_labels))
    #deltas = [_calc_slowness(s=df_agg.get_column("embedding"), dims=n) for n in num_dims]
    #deltas_mock = [_calc_slowness(s=df_agg_mock.get_column("embedding"), dims=n) for n in num_dims]
    #plt.plot(num_dims, deltas)
    #plt.plot(num_dims, deltas_mock)
    #plt.show()


def _calc_slowness_gains_for_embeddings(embeddings: Series, labels: List[str], n_iterations: int) -> DataFrame:
    result = DataFrame()
    for i in tqdm(range(n_iterations)):
        shuffled_labels, shuffled_embeddings = _shuffle_labels_and_embedding_columns(labels, np.array(embeddings))
        delta_diffs = _calc_delta_diffs(shuffled_embeddings)
        df = DataFrame({
            "label": shuffled_labels,
            f"delta_diff_{i}": delta_diffs,
        })
        result = result.with_columns(
            df.sort(by="label")
        )
    return result.with_columns(
        result.select(pl.selectors.starts_with("de")).mean_horizontal().alias("mean_delta_diff"),
    ).select(
        ["label", "mean_delta_diff"]
    ).sort(
        by="mean_delta_diff", descending=True
    )


def _shuffle_labels_and_embedding_columns(labels: List[str], embeddings: np.ndarray) -> Tuple[List[str], np.ndarray]:
    shuffled_idc = generator.permutation(np.arange(len(labels)))
    return (
        [labels[i] for i in shuffled_idc],
        embeddings[:,shuffled_idc],
    )


def _calc_delta_diffs(embeddings: np.ndarray):
    n_dims = embeddings.shape[1]
    deltas = [_calc_slowness(embeddings=embeddings, dims=n) for n in np.arange(1, n_dims+1)]
    return np.array([2.0] + deltas[:-1]) - np.array(deltas)


def _calc_slowness(embeddings: np.ndarray, dims: int) -> float:
    return calc_slowness_for_array(embeddings[:,:dims], dims=2)


if __name__ == "__main__":
    main()
