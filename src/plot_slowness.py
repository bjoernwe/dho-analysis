import matplotlib.pyplot as plt
import numpy as np
from polars import Series

from sklearn.decomposition import PCA
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

    # "empathetic" dataset
    "afraid", "anxious", "apprehensive", "ashamed", "caring",
    "content", "disgusted", "excited", "faithful", "grateful",
    "guilty", "hopeful", "joyful", "lonely",
    #"sentimental", "confident", "terrified", "disappointed", "furious", "surprised", "angry", "impressed", "annoyed",
    #"jealous", "devastated", "anticipating", "trusting", "nostalgic", "prepared", "proud", "sad", "embarrassed",

    # misc
    "pain",
    #"fire",
    "calmness", "spaciousness", "harmony",
    "sensory", "visual", "somatic", "mental",
    "vague", "abstract", "metaphorical", "measurable",
    "passivity",
    "unfamiliar",

    # empirically useless labels:
    # "dissonance", "concrete", "auditory", "equanimity",
    # "familiar", "happiness", "passivity", "satisfaction", "surprising", "agency", "sadness", "paradox", "specific", "confusion"
]


def main():
    #model = SentenceTransformerModel("all-MiniLM-L6-v2", batch_size=1000)
    #model = ClassificationTransformer(model="SamLowe/roberta-base-go_emotions", batch_size=100)
    model = ZeroShotEmbeddingTransformer(model="MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33", labels=zeroshot_labels, batch_size=1000)
    plot_slowness(model=model)


def plot_slowness(
        model: EmbeddingModelABC = SentenceTransformerModel("all-MiniLM-L6-v2"),
        author: str = "Linda ”Polly Ester” Ö",
        time_aggregate: str = "1d",
        pca_components: int = .9,
        sfa_component: int = 0,
):

    # Load practice logs
    df0 = load_practice_logs_for_author(author=author)
    df0 = df0.sort("date")

    # Calc embedding for each sentence in logs
    df_sen = explode_msg_to_sentences(df=df0)
    df_sen = add_message_embeddings(df=df_sen, model=model)

    # Calc PCA for embeddings
    pca = PCA(n_components=pca_components, random_state=SEED)
    pca.fit(np.array(df_sen["embedding"]))

    # Apply PCA to embeddings
    df_sen = df_sen.with_columns(Series("embedding", pca.transform(np.array(df_sen["embedding"]))))

    # Aggregate messages and embeddings time-wise
    df_agg = aggregate_messages_by_time(df=df_sen.select(["date", "msg", "embedding"]), time_aggregate=time_aggregate)

    # Calc SFA for embeddings
    sfa = SFA(n_components=sfa_component+1, random_state=SEED)
    sfa.fit(np.array(df_agg["embedding"]))

    # Apply SFA to embeddings
    df_sen = df_sen.with_columns(Series("SFA", sfa.transform(np.array(df_sen["embedding"]))[:,sfa_component]))
    df_agg = df_agg.with_columns(Series("SFA", sfa.transform(np.array(df_agg["embedding"]))[:,sfa_component]))

    # Print most representative sentences
    for s in df_sen.sort("SFA")["msg"].to_list()[-10:]: print(s)
    print("\n...\n")
    for s in df_sen.sort("SFA")["msg"].to_list()[:10][::-1]: print(s)

    # Plot slowest feature
    plt.figure(figsize=(10, 5))
    plt.plot(df_agg.select(["date"]), df_agg.select(["SFA"]))
    plt.show()


if __name__ == "__main__":
    main()
