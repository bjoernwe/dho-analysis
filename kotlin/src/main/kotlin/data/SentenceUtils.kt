package data

import org.jetbrains.kotlinx.dataframe.DataFrame

fun readRawSentences(): List<String> {
    return readSentences().map { it.sentence }
}

fun readSentences(): List<Sentence> = readMessages().getSentences()

fun DataFrame<Message>.getSentences(): List<Sentence> {
    return this.sentences.toList().flatten()
}

fun List<Sentence>.sortByDescendingLength(): List<Sentence> {
    return this.sortedByDescending { it.sentence.length }
}

fun List<Sentence>.rawStrings(): List<String> {
    return this.map { it.sentence }
}
