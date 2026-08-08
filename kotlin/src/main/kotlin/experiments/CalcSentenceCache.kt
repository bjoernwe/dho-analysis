package experiments

import data.getSentences
import data.rawStrings
import data.readMessages
import me.tongfei.progressbar.ProgressBar
import models.defaultModel

fun main() {

    val sentences = readMessages()
        .filterByLindasPracticeLogs()
        .getSentences()

    println("Scoring ${sentences.size} sentences...")

    defaultModel.use { model ->
        for (label in labels) {
            println(label)
            ProgressBar("Scoring", (sentences.size).toLong()).use { progressBar ->
                model.score(sentences.rawStrings(), label) { n -> progressBar.stepBy(n.toLong()) }
            }
        }
    }
}
