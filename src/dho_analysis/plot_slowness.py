import matplotlib.pyplot as plt

from dho_analysis.calc_message_embeddings import add_message_embeddings
from dho_analysis.calc_slow_features import add_sfa_columns
from dho_analysis.load_time_aggregated_practice_logs import load_time_aggregated_practice_logs


def main():
    plot_slowness()


def plot_slowness():
    df = load_time_aggregated_practice_logs(time_aggregate="1mo", author="Linda ”Polly Ester” Ö")
    print(df)
    df = add_message_embeddings(df)
    df = add_sfa_columns(df, pca=.9)
    plt.plot(df.select(["date"]), df.select(["SFA_0"]))
    plt.show()


if __name__ == "__main__":
    main()
