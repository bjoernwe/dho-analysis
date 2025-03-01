import numpy as np
import sksfa
from polars import Series
from sklearn.decomposition import PCA

from config import SEED
from data.load_time_aggregated_practice_logs import load_time_aggregated_practice_logs
from models.SentenceTransformerModel import SentenceTransformerModel


def main():
    print_example_slowness_value()


def print_example_slowness_value():
    model = SentenceTransformerModel("all-MiniLM-L6-v2")
    df = load_time_aggregated_practice_logs(time_aggregate="1w", author="Linda ”Polly Ester” Ö", model=model)
    delta = calc_slowness(df.get_column("embedding"))
    print(delta)


def calc_slowness(x: Series) -> float:
    components = PCA(n_components=.99, random_state=SEED).fit_transform(np.array(x))
    sfa = sksfa.SFA(n_components=1)
    sfa.fit(components)
    return float(sfa.delta_values_[0])


if __name__ == "__main__":
    main()
