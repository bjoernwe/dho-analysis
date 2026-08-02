package experiments

import data.readRawSentences
import me.tongfei.progressbar.ProgressBar
import models.defaultModel

fun main() {

    val sentences = readRawSentences().sortedBy { it.length }
    println("Scoring ${sentences.size} sentences...")

    defaultModel.use { model ->
        for (label in labels) {
            println(label)
            ProgressBar("Scoring", (sentences.size).toLong()).use { progressBar ->
                model.score(sentences, label) { n -> progressBar.stepBy(n.toLong()) }
            }
        }
    }
}
