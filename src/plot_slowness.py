import matplotlib.pyplot as plt
import numpy as np
from polars import Series

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


zeroshot_labels = [
    "turtles",  # as baseline

    "positive", "negative",
    "uncertainty", "certainty",
    "past", #"future", #"present",

    # "empathetic" dataset
    "sentimental", "impressed", "joyful", "excited",
    "feeling content", #"being prepared",
    #"jealous", "guilty", "embarrassed", "ashamed", "nostalgic", "lonely", "afraid", "annoyed", "terrified", "proud",
    #"angry", "devastated", "caring", "apprehensive", "furious", "disgusted", "anxious", "sad", "surprised",
    "confident",
    #"disappointed", "faithful", "grateful", "trusting", "hopeful", "anticipating",

    # misc
    #"fire",
    "sensory", "visual", "somatic", "mental",
    "vague", "abstract", "measurable",
    "passivity", "calmness", "harmony",
    "equanimity", "metaphorical",
    "dissonance", "auditory",
    "familiar", "happiness", "passivity", "agency", "paradox", "specific",
    "confusion", #"unfamiliar",
    # "concrete",
    "pain", #"sadness", "satisfaction",
    "spaciousness",
]


def main():
    #model = SentenceTransformerModel("all-MiniLM-L6-v2", batch_size=1000)
    #model = ClassificationTransformer(model="SamLowe/roberta-base-go_emotions", batch_size=100)
    model = ZeroShotEmbeddingTransformer(model="MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33", labels=zeroshot_labels, batch_size=1000)
    plot_slowness(
        model=model,
        author="Linda ”Polly Ester” Ö",
        #author="Siavash '",
        #author="Papa Che Dusko",
        #author="George S",
        #author="Sam Gentile",
        #author="Noah",
    )


def plot_slowness(
        model: EmbeddingModelABC = SentenceTransformerModel("all-MiniLM-L6-v2"),
        author: str = "Linda ”Polly Ester” Ö",
        time_aggregate: str = "1d",
        pca_min_explained: float = 3e-2,
        sfa_component: int = 1,
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
    sfa = SFA(n_components=3, robustness_cutoff=pca_min_explained, fill_mode='zero', random_state=SEED)
    sfa.fit(np.array(df_agg["embedding"]))

    # Apply SFA to embeddings
    df_sen = df_sen.with_columns(Series("SFA", sfa.transform(np.array(df_sen["embedding"]))[:,sfa_component]))
    df_agg = df_agg.with_columns(Series("SFA", sfa.transform(np.array(df_agg["embedding"]))[:,sfa_component]))

    # Print most representative sentences
    for s in df_sen.sort("SFA")["msg"].to_list()[-10:]: print(s)
    print("\n...\n")
    for s in df_sen.sort("SFA")["msg"].to_list()[:10][::-1]: print(s)

    # Print PCA info
    print(f"\nPCA: {sfa.input_dim_} -> {sfa.n_nontrivial_components_}")
    #print(f"Explained variance: {sfa.pca_whiten_.explained_variance_}")
    #print(f"Delta values: {sfa.delta_values_[:sfa.n_nontrivial_components_]}\n")

    # Plots
    plot_pca_and_sfa_variances(sfa=sfa)
    plot_temporal_label_importance(sfa=sfa, labels=zeroshot_labels)
    plot_sfa_weights(sfa=sfa, component=0)
    plot_sfa_weights(sfa=sfa, component=1)

    # Plot slowest feature
    plt.figure()
    plt.plot(df_agg.select(["date"]), df_agg.select(["SFA"]))
    plt.show()


def plot_pca_and_sfa_variances(sfa: SFA):
    #plt.figure()
    fig, ax1 = plt.subplots()
    plt.title("Explained variance / delta")
    ax1.plot(sfa.pca_whiten_.explained_variance_, label="variance")
    ax2 = plt.twinx()
    ax2.plot(sfa.delta_values_, label="delta")


def plot_sfa_weights(sfa: SFA, component: int = 0):

    sfa_weights_unsorted = sfa.affine_parameters()[0][component]
    idc = np.argsort(sfa_weights_unsorted)
    sfa_weights = sfa_weights_unsorted[idc]

    color_limit = max(abs(min(sfa_weights)), abs(max(sfa_weights)))
    plt.figure()
    plt.title(f"Label weights in SFA component #{component}")
    plt.barh(
        [zeroshot_labels[i] for i in idc],
        sfa_weights,
        color=plt.get_cmap("PiYG")(plt.Normalize(-color_limit, color_limit)(sfa_weights))
    )


def plot_temporal_label_importance(sfa: SFA, labels: list[str]):

    weighted_weights = sfa.affine_parameters()[0] / sfa.delta_values_[:sfa.n_components, np.newaxis]
    label_importance = np.max(np.abs(weighted_weights), axis=0)
    idc = np.argsort(label_importance)
    pca_weights = np.sum(np.abs(sfa.pca_whiten_.get_covariance()), axis=0)[idc]

    plt.figure()
    plt.title("Temporal importance per label (color: PCA weight)")
    plt.barh(
        [labels[i] for i in idc],
        label_importance[idc],
        color=plt.get_cmap("Blues")(plt.Normalize(0, max(pca_weights))(pca_weights))
    )


if __name__ == "__main__":
    main()
