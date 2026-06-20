package experiments

import data.readMessages
import kotlin.io.path.Path
import models.CachingZeroShotClassifier
import models.OnnxZeroShotClassifier
import models.OpenNlpSentenceSplitter
import org.jetbrains.kotlinx.dataframe.DataColumn

fun main() {
    val splitter = OpenNlpSentenceSplitter()
    val messages = readMessages(splitter)
    @Suppress("UNCHECKED_CAST")
    val sentences = (messages["sentences"] as DataColumn<List<String>>).toList().flatten()

    val modelName = "ModernBERT-large-zeroshot-v2.0"
    val delegate = OnnxZeroShotClassifier(
        modelDir = Path("models/MoritzLaurer").resolve(modelName),
        modelFile = "model_fp16.onnx"
    )

    val labels = listOf("positive", "negative", "neutral")

    CachingZeroShotClassifier(delegate, Path("cache/scores.db"), modelName).use { model ->
        for (label in labels) {
            val scores = model.scoreBatch(sentences, label)
            for ((sentence, score) in sentences.zip(scores)) {
                println("%-80s %-12s %.3f".format(sentence, label, score))
            }
        }
    }
}
