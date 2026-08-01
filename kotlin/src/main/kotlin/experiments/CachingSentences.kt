package experiments

import data.readSentences
import me.tongfei.progressbar.ProgressBar
import models.defaultModel

fun main() {

    val sentences = readSentences()
    println("Scoring ${sentences.size} sentences...")

    defaultModel.use { model ->
        for (label in labels) {
            println(label)
            ProgressBar("Scoring", (sentences.size).toLong()).use { progressBar ->
                for (batch in createBatches(sentences)) {
                    model.scoreBatch(batch, label)
                    progressBar.stepBy(batch.size.toLong())
                }
            }
        }
    }
}
