from typing import List

import matplotlib.pyplot as plt
import numpy as np
from polars import Series, DataFrame

from sksfa import SFA

from functions.calc_message_embeddings import add_message_embeddings
from functions.calc_sentences import explode_msg_to_sentences
from data.load_practice_logs_for_author import load_practice_logs_for_author
from data.load_time_aggregated_practice_logs import aggregate_messages_by_time
from models.ClassificationTransformer import ClassificationTransformer
from models.EmbeddingModelABC import EmbeddingModelABC
from config import SEED
from models.SentenceTransformerModel import SentenceTransformerModel
from models.ZeroShotEmbeddingTransformer import ZeroShotEmbeddingTransformer


zeroshot_labels: List[str] = [
    "turtles",  # as baseline
    "greeting",

    "positive", #"negative",
    "uncertainty", #"certainty",
    "past", #"future", #"present",

    # "empathetic" dataset
    "sentimental", "impressed", "excited", #"joyful",
    "feeling content", #"being prepared",
    #"jealous", "guilty", "embarrassed", "ashamed", "nostalgic", "lonely", "afraid", "annoyed", "terrified", "proud",
    #"angry", "devastated", "caring", "apprehensive", "furious", "disgusted", "anxious", "sad", "surprised",
    #"confident",
    #"disappointed", "faithful", "grateful", "trusting", "hopeful", "anticipating",

    # misc
    "fire",
    "meditation",
    "sensory", "visual", "somatic", "mental",
    "vague", "abstract", "measurable",
    "passivity", "calmness", "harmony",
    "equanimity", "metaphorical",
    "dissonance", "auditory",
    "passivity", "paradox", "specific", #"agency", "happiness",
    "confusion", #"familiar", "unfamiliar",
    # "concrete",
    #"pain", #"sadness", "satisfaction",
    "space",
    "conceptual", "non-conceptual",
    "tingling", "vibrations",
    "sleepiness", "alertness", "dullness", "energetic",
    "direct experience",
    "weird",
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
        time_aggregate: str = "1d",
        pca_min_explained: float = 3e-2,
        n_sfa_components: int = 3,
):

    # Load practice logs
    df0 = load_practice_logs_for_author(author=author)
    df0 = df0.sort("date")

    # Calc embedding for each sentence in logs
    df_sen = explode_msg_to_sentences(df=df0)
    df_sen = add_message_embeddings(df=df_sen, model=model)

    # Aggregate messages and embeddings time-wise
    df_agg = aggregate_messages_by_time(df=df_sen.select(["date", "msg", "embedding"]), time_aggregate=time_aggregate)

    # Calc SFA for embeddings
    sfa = SFA(n_components=n_sfa_components, robustness_cutoff=pca_min_explained, fill_mode='zero', random_state=SEED)
    sfa.fit(np.array(df_agg["embedding"]))

    # Apply SFA to embeddings
    df_sen = add_sfa_from_embedding(df=df_sen, sfa=sfa, n_components=n_sfa_components)
    df_agg = add_sfa_from_embedding(df=df_agg, sfa=sfa, n_components=n_sfa_components)

    # Print most representative sentences
    print_pca_sentences(df_sen=df_sen, sfa=sfa, n_pca_components=4)
    print_sfa_sentences(df_sen=df_sen, n_sfa_components=n_sfa_components)

    # Plots
    plot_explained_variances(sfa=sfa)
    plot_temporal_label_importance(sfa=sfa, labels=zeroshot_labels, df=df_agg)
    plot_pca_weights(sfa=sfa, labels=zeroshot_labels, dim=0)
    plot_pca_weights(sfa=sfa, labels=zeroshot_labels, dim=1)
    plot_pca_weights(sfa=sfa, labels=zeroshot_labels, dim=2)
    plot_pca_weights(sfa=sfa, labels=zeroshot_labels, dim=3)
    plot_sfa_weights(sfa=sfa, component=0)
    plot_sfa_weights(sfa=sfa, component=1)
    plot_sfa_weights(sfa=sfa, component=2)

    # Plot slowest feature
    plt.figure()
    plt.plot(df_agg.select(["date"]), df_agg.select(["SFA_0"]))
    plt.show()


def add_sfa_from_embedding(df: DataFrame, sfa: SFA, n_components: int):
    sfa_features = sfa.transform(np.array(df["embedding"]))
    result = DataFrame(df)
    for i in range(n_components):
        result = result.with_columns(Series(f"SFA_{i}", sfa_features[:, i]))
    return result


def print_pca_sentences(df_sen: DataFrame, sfa: SFA, n_pca_components: int):
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


def plot_explained_variances(sfa: SFA):
    plt.figure()
    plt.title(f"PCA: Explained variance ({sfa.input_dim_} => {sfa.n_nontrivial_components_})")
    plt.bar(
        np.arange(sfa.input_dim_),
        sfa.pca_whiten_.explained_variance_,
        color=['royalblue'] * sfa.n_nontrivial_components_ + ['deepskyblue'] * (sfa.input_dim_ - sfa.n_nontrivial_components_),
        label="variance",
    )


def plot_pca_weights(sfa: SFA, labels: List[str], dim: int = 0):

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


def plot_sfa_weights(sfa: SFA, component: int = 0):

    sfa_weights_unsorted = sfa.affine_parameters()[0][component]
    idc = np.argsort(sfa_weights_unsorted)
    sfa_weights = sfa_weights_unsorted[idc]

    plt.figure()
    plt.title(f"Label weights in SFA component #{component}")
    color_limit = max(abs(min(sfa_weights)), abs(max(sfa_weights)))
    plt.barh(
        [zeroshot_labels[i] for i in idc],
        sfa_weights,
        color=plt.get_cmap("PiYG")(plt.Normalize(-color_limit, color_limit)(sfa_weights))
    )


def plot_temporal_label_importance(sfa: SFA, labels: list[str], df: DataFrame):

    normalized_components = sfa.affine_parameters()[0] / np.linalg.norm(sfa.affine_parameters()[0], axis=1)[:, np.newaxis]
    weighted_components = normalized_components / (2 - sfa.delta_values_).clip(min=0)[:sfa.n_components, np.newaxis]
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


if __name__ == "__main__":
    main()
