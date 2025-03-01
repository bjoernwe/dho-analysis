import matplotlib.pyplot as plt

from functions.calc_slowness_value import calc_slowness
from data.load_time_aggregated_practice_logs import load_time_aggregated_practice_logs
from models.EmbeddingModelABC import EmbeddingModelABC
from models.SentenceTransformerModel import SentenceTransformerModel


def main():
    plot_slowness_for_different_time_scales()


def plot_slowness_for_different_time_scales():
    model = SentenceTransformerModel("all-MiniLM-L6-v2")
    time_aggregates = [f"{i}d" for i in range(1, 31)]
    deltas = [_calc_slowness_for_time_aggregate(model=model, time_aggregate=ta) for ta in time_aggregates]
    plt.plot(deltas)
    print(deltas)
    plt.show()


def _calc_slowness_for_time_aggregate(model: EmbeddingModelABC, time_aggregate: str = "1d"):
    df = load_time_aggregated_practice_logs(time_aggregate=time_aggregate, author="Linda ”Polly Ester” Ö", model=model)
    embeddings = df.get_column("embedding")
    return calc_slowness(embeddings)


if __name__ == "__main__":
    main()
