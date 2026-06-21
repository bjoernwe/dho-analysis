package experiments

import data.readMessages
import kotlin.io.path.Path
import me.tongfei.progressbar.ProgressBar
import models.CachingZeroShotClassifier
import models.OnnxZeroShotClassifier
import org.jetbrains.kotlinx.dataframe.DataColumn

const val BATCH_SIZE_TOKEN_BUDGET = 45_000

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
                for (batch in batchByLengthBudget(sentences, maxBudget = BATCH_SIZE_TOKEN_BUDGET)) {
                    model.scoreBatch(batch, label)
                    progressBar.stepBy(batch.size.toLong())
                }
            }
        }
    }
}

// Greedily groups already-length-sorted sentences so that batchSize * maxLength stays under
// maxBudget, keeping the per-batch GPU activation memory roughly constant regardless of sentence length.
fun batchByLengthBudget(sortedSentences: List<String>, maxBudget: Int): List<List<String>> {
    val batches = mutableListOf<List<String>>()
    var batch = mutableListOf<String>()
    for (sentence in sortedSentences) {
        if (batch.isNotEmpty() && (batch.size + 1) * sentence.length > maxBudget) {
            batches.add(batch)
            batch = mutableListOf()
        }
        batch.add(sentence)
    }
    if (batch.isNotEmpty()) batches.add(batch)
    return batches
}
