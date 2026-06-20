package experiments

import data.readMessages
import kotlin.io.path.Path
import me.tongfei.progressbar.ProgressBar
import models.CachingZeroShotClassifier
import models.OnnxZeroShotClassifier
import org.jetbrains.kotlinx.dataframe.DataColumn

fun main() {

    val messages = readMessages()
    @Suppress("UNCHECKED_CAST")
    val sentences = (messages["sentences"] as DataColumn<List<String>>).toList().flatten().sortedBy { it.length }

    val modelName = "ModernBERT-large-zeroshot-v2.0"
    val delegate = OnnxZeroShotClassifier(
        modelDir = Path("models/MoritzLaurer").resolve(modelName),
        modelFile = "model_fp16.onnx"
    )

    val labels = listOf("positive", "negative", "neutral")

    CachingZeroShotClassifier(delegate, Path("cache/scores.db"), modelName).use { model ->
        ProgressBar("Scoring", (labels.size * sentences.size).toLong()).use { progressBar ->
            for (label in labels) {
                for (batch in sentences.chunked(1_000)) {
                    val scores = model.scoreBatch(batch, label)
                    /*for ((sentence, score) in batch.zip(scores)) {
                        println("%-12s %.3f %-80s".format(label, score, sentence))
                    }*/
                    progressBar.stepBy(batch.size.toLong())
                }
            }
        }
    }
}
