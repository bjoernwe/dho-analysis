from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import FuncFormatter

from polars import Series, DataFrame
from scipy.fft import fft, fftfreq
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sksfa import SFA

from functions.calc_message_embeddings import add_message_embeddings
from functions.calc_sentences import explode_msg_to_sentences
from data.load_practice_logs_for_author import load_practice_logs_for_author
from data.load_time_aggregated_practice_logs import aggregate_messages_by_time
from models.ClassificationTransformer import ClassificationTransformer
from models.CustomPCA import CustomPCA
from models.EmbeddingModelABC import EmbeddingModelABC
from config import SEED
from models.SentenceTransformerModel import SentenceTransformerModel
from models.ZeroShotEmbeddingTransformer import ZeroShotEmbeddingTransformer


zeroshot_labels: List[str] = [
    "turtles",  # sanity check
    "greeting",

    "positive", "negative",
    "uncertainty", #"certainty",
    #"past", "present", #"future",

    # "empathetic" dataset
    "sentimental", "impressed", "excited", #"joyful",
    "feeling content", #"being prepared",
    #"jealous", "guilty", "embarrassed", "ashamed", "nostalgic", "lonely", "afraid", "annoyed", "terrified", "proud",
    #"angry", "devastated", "caring", "apprehensive", "furious", "disgusted", "anxious", "sad", "surprised",
    "confident",
    "anticipating", "grateful", #"disappointed", "faithful", "trusting", "hopeful",

    # misc
    #"fire", "yoga", "jhana",
    "meditation",
    "body", "mind",
    "high concentration", "low concentration",
    "sensory", "visual", "somatic", "mental",
    "vague", "abstract", "measurable",
    "passivity", "calmness", "harmony",
    "equanimity", "metaphorical",
    "dissonance", "auditory",
    "passivity", "paradox", "specific", #"agency", "happiness",
    "confusion", #"familiar", "unfamiliar",
    # "concrete",
    "pain", #"sadness", "satisfaction",
    "space",
    "subjective", "objective",
    "conceptual", "non-conceptual",
    "tingling", "vibrations",
    "sleepiness", "alertness", "dullness", "energetic",
    "direct experience",
    "weird", "struggling", "being challenged",
    #"necessity", "options",
    "owning sth",
]


def main():

    #model = SentenceTransformerModel("all-MiniLM-L6-v2", batch_size=1000)
    #model = ClassificationTransformer(model="SamLowe/roberta-base-go_emotions", batch_size=100)
    model = ZeroShotEmbeddingTransformer(model="MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33", labels=zeroshot_labels, batch_size=1000)

    author = "Linda ”Polly Ester” Ö"
    #author = "Siavash '"
    #author = "Papa Che Dusko"
    #author = "George S"
    #author = "Sam Gentile"
    #author = "Noah"

    plot_slowness(
        model=model,
        author=author,
    )


def plot_slowness(
        model: EmbeddingModelABC = SentenceTransformerModel("all-MiniLM-L6-v2"),
        author: str = "Linda ”Polly Ester” Ö",
        time_aggregate_period_train: Optional[str] = "1w",
        time_aggregate_period_plot: Optional[str] = "1d",
        pca_min_explained: float = 2e-2,
        n_pca_components: int = 4,
        n_sfa_components: int = 3,
):

    # Load practice logs
    df0 = load_practice_logs_for_author(author=author)
    df0 = df0.sort("date")

    # Calc embedding for each sentence in logs
    df_sen = explode_msg_to_sentences(df=df0)
    df_sen = add_message_embeddings(df=df_sen, model=model)

    # Aggregate messages and embeddings time-wise
    df_train = aggregate_messages_by_time(
        df=df_sen.select(["date", "msg", "embedding"]),
        every='1d',
        period=time_aggregate_period_train,
    )[:800]
    df_plot = aggregate_messages_by_time(
        df=df_sen.select(["date", "msg", "embedding"]),
        every='1d',
        period=time_aggregate_period_plot,
    )#[:100]

    # Calc PCA for sentences
    pca = CustomPCA(n_components=n_pca_components)
    pca.fit(np.array(df_sen["embedding"]))

    # Apply PCA to embeddings
    df_sen = apply_pca_to_embedding(df=df_sen, pca=pca)
    df_train = apply_pca_to_embedding(df=df_train, pca=pca)
    df_plot = apply_pca_to_embedding(df=df_plot, pca=pca)

    # Calc SFA for embeddings
    sfa = SFA(n_components=n_sfa_components, robustness_cutoff=pca_min_explained, fill_mode='zero', random_state=SEED)
    sfa.fit(np.array(df_train["embedding"]))

    # Apply SFA to embeddings
    df_sen = add_sfa_from_embedding(df=df_sen, sfa=sfa, n_components=n_sfa_components)
    df_train = add_sfa_from_embedding(df=df_train, sfa=sfa, n_components=n_sfa_components)
    df_plot = add_sfa_from_embedding(df=df_plot, sfa=sfa, n_components=n_sfa_components)

    # Print most representative sentences
    print_pca_sentences(df_sen=df_sen, n_components=n_pca_components)
    print_sfa_sentences(df_sen=df_sen, n_sfa_components=n_sfa_components)
    print(sfa.delta_values_[:n_sfa_components])

    # Plots
    plot_explained_variance_from_pca(pca=pca)
    for i in range(n_pca_components):
        plot_labels_for_pca_component(pca=pca, labels=zeroshot_labels, component=i)
    for i in range(n_sfa_components):
        plot_pca_weights_in_sfa(sfa=sfa, component=i)
    for i in range(n_sfa_components):
        plot_labels_for_sfa_component(sfa=sfa, pca=pca, labels=zeroshot_labels, component=i)
    for i in range(n_sfa_components):
        plot_fft(df=df_plot, component=i)

    # Plot slowest feature
    for i in range(n_sfa_components):
        plt.figure()
        plt.title(f"SFA #{i}")
        #plt.plot(df_plot.select(["date"]), df_plot.select([f"SFA_{i}"]), color="deepskyblue", zorder=0)
        plt.plot(df_train.select(["date"]), df_train.select([f"SFA_{i}"]), color="silver", zorder=1)
        plt.scatter(df_train.select(["date"]), df_train.select([f"SFA_{i}"]), c=df_train.select(["SFA_2"]), cmap="PiYG", s=40, alpha=.8, zorder=2)
    #plot_gaussian_process(df=df_agg)
    plt.show()


def apply_pca_to_embedding(df: DataFrame, pca: CustomPCA):
    pca_features = pca.transform(np.array(df["embedding"]))
    result = df.with_columns(
        Series("embedding", pca_features)
    )
    return result


def add_sfa_from_embedding(df: DataFrame, sfa: SFA, n_components: int):
    sfa_features = sfa.transform(np.array(df["embedding"]))
    result = DataFrame(df)
    for i in range(n_components):
        result = result.with_columns(Series(f"SFA_{i}", sfa_features[:, i]))
    return result


def print_pca_sentences(df_sen: DataFrame, n_components: int):
    pca_features = np.array(df_sen["embedding"])
    for i in range(n_components):
        print("******************************")
        print(f"Sentences for PCA component #{i}")
        print("******************************\n")
        df_sen_pca = df_sen.with_columns(Series(f"PCA_{i}", pca_features[:, i]))
        for s in df_sen_pca.sort(f"PCA_{i}")["msg"].to_list()[-10:]: print(s)
        print("\n...\n")
        for s in df_sen_pca.sort(f"PCA_{i}")["msg"].to_list()[:10][::-1]: print(s)
        print("\n")


def print_pca_sentences_from_sfa(df_sen: DataFrame, sfa: SFA, n_pca_components: int):
    pca_features = sfa.pca_whiten_.transform(np.array(df_sen["embedding"]))
    for i in range(n_pca_components):
        print("******************************")
        print(f"Sentences for PCA component #{i}")
        print("******************************\n")
        df_sen_pca = df_sen.with_columns(Series(f"PCA_{i}", pca_features[:, i]))
        for s in df_sen_pca.sort(f"PCA_{i}")["msg"].to_list()[-10:]: print(s)
        print("\n...\n")
        for s in df_sen_pca.sort(f"PCA_{i}")["msg"].to_list()[:10][::-1]: print(s)
        print("\n")


def print_sfa_sentences(df_sen: DataFrame, n_sfa_components: int):
    for i in range(n_sfa_components):
        print("******************************")
        print(f"Sentences for SFA component #{i}")
        print("******************************\n")
        for s in df_sen.sort(f"SFA_{i}")["msg"].to_list()[-10:]: print(s)
        print("\n...\n")
        for s in df_sen.sort(f"SFA_{i}")["msg"].to_list()[:10][::-1]: print(s)
        print("\n")


def plot_explained_variance_from_pca(pca: CustomPCA):
    output_dim, input_dim = pca.components_reduced_.shape
    plt.figure()
    plt.title(f"PCA: Explained variance ({input_dim} => {output_dim})")
    plt.bar(
        np.arange(input_dim),
        pca.explained_variance_ratio_full_,
        color=['royalblue'] * output_dim + ['deepskyblue'] * (input_dim - output_dim),
        label="variance",
    )


def plot_explained_variance_from_sfa(sfa: SFA):
    plt.figure()
    plt.title(f"PCA: Explained variance ({sfa.input_dim_} => {sfa.n_nontrivial_components_})")
    plt.bar(
        np.arange(sfa.input_dim_),
        sfa.pca_whiten_.explained_variance_,
        color=['royalblue'] * sfa.n_nontrivial_components_ + ['deepskyblue'] * (sfa.input_dim_ - sfa.n_nontrivial_components_),
        label="variance",
    )


def plot_labels_for_pca_component(pca: CustomPCA, labels: List[str], component: int = 0):

    weights_unsorted = pca.components_reduced_[component]
    idc = np.argsort(weights_unsorted)
    weights = weights_unsorted[idc]

    plt.figure()
    plt.title(f"Label weights in PCA component #{component}")
    color_limit = max(abs(min(weights)), abs(max(weights)))
    plt.barh(
        np.array(labels)[idc],
        weights,
        color=plt.get_cmap("RdBu")(plt.Normalize(-color_limit, color_limit)(weights))
    )


def plot_labels_for_sfa_component(sfa: SFA, pca: CustomPCA, labels: List[str], component: int = 0):

    pca_weights = pca.components_reduced_
    sfa_weights = sfa.affine_parameters()[0]
    weights_unsorted = sfa_weights.dot(pca_weights)[component]
    idc = np.argsort(weights_unsorted)
    weights = weights_unsorted[idc]

    plt.figure()
    plt.title(f"Label weights in SFA component #{component}")
    color_limit = max(abs(min(weights)), abs(max(weights)))
    plt.barh(
        np.array(labels)[idc],
        weights,
        color=plt.get_cmap("PiYG")(plt.Normalize(-color_limit, color_limit)(weights))
    )


def plot_pca_weights_from_sfa(sfa: SFA, labels: List[str], dim: int = 0):

    weights_unsorted = sfa.pca_whiten_.components_[dim]
    idc = np.argsort(weights_unsorted)
    weights = weights_unsorted[idc]

    plt.figure()
    plt.title(f"Label weights in PCA component #{dim}")
    color_limit = max(abs(min(weights)), abs(max(weights)))
    plt.barh(
        np.array(labels)[idc],
        weights,
        color=plt.get_cmap("RdBu")(plt.Normalize(-color_limit, color_limit)(weights))
    )


def plot_pca_weights_in_sfa(sfa: SFA, component: int = 0):

    sfa_weights = sfa.affine_parameters()[0][component]

    _, ax = plt.subplots()
    ax.set_title(f"PCA weights in SFA component #{component}")
    color_limit = max(abs(min(sfa_weights)), abs(max(sfa_weights)))
    labels = [f"PCA #{i}" for i in range(sfa.input_dim_)]
    ax.barh(
        labels,
        sfa_weights,
        color=plt.get_cmap("bwr")(plt.Normalize(-color_limit, color_limit)(sfa_weights))
    )
    ax.invert_yaxis()


def plot_temporal_label_importance(sfa: SFA, labels: list[str], df: DataFrame):

    normalized_components = sfa.affine_parameters()[0] / np.linalg.norm(sfa.affine_parameters()[0], axis=1)[:, np.newaxis]
    weighted_components = normalized_components * (2 - sfa.delta_values_).clip(min=0)[:sfa.n_components, np.newaxis]
    label_importance = np.max(np.abs(weighted_components), axis=0)
    idc = np.argsort(label_importance)
    label_variances = np.var(np.array(df['embedding']), axis=0)[idc]

    plt.figure()
    plt.title("Temporal importance per label (color: variance)")
    plt.barh(
        [labels[i] for i in idc],
        label_importance[idc],
        color=plt.get_cmap("Blues")(plt.Normalize(0, max(label_variances))(label_variances))
    )


def plot_fft(df: DataFrame, component: int = 0):

    y = np.array(df.select([f"SFA_{component}"]))[:,0]
    N = y.shape[0]
    yf = fft(y)
    xf = fftfreq(N, 1)[:N//2]

    def x_labels(x, pos):
        return f'{1 / x:.2f}'

    _, ax = plt.subplots()
    ax.xaxis.set_major_formatter(FuncFormatter(x_labels))
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))


def plot_gaussian_process(
        df: DataFrame,
        length_scale: float = 50.0,
        alpha: float = 1,
        sigma: float = 1,
        length_scale_bounds: Tuple[float, float]=(10, 100),
):

    X_train = np.array(df["date"].cast(pl.Int64)) / (1000*1000*60*60*24)
    y_train = np.array(df["SFA_0"])
    kernel = sigma * RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
    gaussian_process = GaussianProcessRegressor(
        kernel=kernel, alpha=alpha, n_restarts_optimizer=9
    )
    gaussian_process.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))
    mean_prediction, std_prediction = gaussian_process.predict(X_train.reshape(-1, 1), return_std=True)

    print(gaussian_process.kernel_)
    plt.plot(X_train, mean_prediction, color="black")
    plt.fill_between(
        X_train.reshape(-1, 1).ravel(),
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        color="tab:orange",
        alpha=0.5,
        label=r"95% confidence interval",
    )


if __name__ == "__main__":
    main()
