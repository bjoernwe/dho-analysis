import matplotlib.pyplot as plt

from dho_analysis.calc_message_embeddings import add_message_embeddings
from dho_analysis.calc_slowness_value import calc_slowness
from dho_analysis.load_time_aggregated_practice_logs import load_time_aggregated_practice_logs


def main():
    plot_slowness_for_different_time_scales()


def plot_slowness_for_different_time_scales():
    time_aggregates = [f"{i}d" for i in range(1, 31)]
    deltas = [_calc_slowness_for_time_aggregate(ta) for ta in time_aggregates]
    plt.plot(deltas)
    print(deltas)
    plt.show()


def _calc_slowness_for_time_aggregate(time_aggregate: str = "1d"):
    df = load_time_aggregated_practice_logs(time_aggregate=time_aggregate)
    df = add_message_embeddings(df)
    embeddings = df.get_column("embedding")
    return calc_slowness(embeddings)


if __name__ == "__main__":
    main()
