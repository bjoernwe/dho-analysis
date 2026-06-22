package experiments

import data.readSentences
import me.tongfei.progressbar.ProgressBar
import models.defaultModel

const val BATCH_SIZE_TOKEN_BUDGET = 47_500

fun main() {

    val sentences = readSentences()
    println("Scoring ${sentences.size} sentences...")

    defaultModel.use { model ->
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
