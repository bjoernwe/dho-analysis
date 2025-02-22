import numpy as np
import sksfa
from polars import Series
from sklearn.decomposition import PCA

from dho_analysis.calc_message_embeddings import add_message_embeddings
from dho_analysis.load_time_aggregated_practice_logs import load_time_aggregated_practice_logs
from dho_analysis.utils import SEED


def main():
    df = load_time_aggregated_practice_logs(time_aggregate="1w", author="Linda ”Polly Ester” Ö")
    df = add_message_embeddings(df=df)
    delta = calc_slowness(df.get_column("embedding"))
    print(delta)


def calc_slowness(x: Series) -> float:
    components = PCA(n_components=.99, random_state=SEED).fit_transform(np.array(x))
    sfa = sksfa.SFA(n_components=1)
    sfa.fit(components)
    return float(sfa.delta_values_[0])


if __name__ == "__main__":
    main()
