from typing import Optional

import matplotlib.pyplot as plt

from functions.calc_slowness_value import calc_slowness_for_series
from data.load_time_aggregated_practice_logs import load_time_aggregated_practice_logs
from models.EmbeddingModelABC import EmbeddingModelABC
from models.SentenceTransformerModel import SentenceTransformerModel


def main():
    plot_slowness_for_different_time_scales()


def plot_slowness_for_different_time_scales():
    model = SentenceTransformerModel("all-MiniLM-L6-v2")
    time_aggregates = [f"{i}d" for i in range(1, 31)]
    deltas = [_calc_slowness_for_time_aggregate(model=model, time_aggregate_every=ta) for ta in time_aggregates]
    plt.plot(deltas)
    print(deltas)
    plt.show()


def _calc_slowness_for_time_aggregate(model: EmbeddingModelABC, time_aggregate_every: str = "1d", time_aggregate_period: Optional[str] = None):
    df = load_time_aggregated_practice_logs(
        author="Linda ”Polly Ester” Ö",
        model=model,
        time_aggregate_every=time_aggregate_every,
        time_aggregate_period=time_aggregate_period,
    )
    embeddings = df.get_column("embedding")
    return calc_slowness_for_series(embeddings)


if __name__ == "__main__":
    main()
