package data

import models.SentenceSplitter
import models.defaultSplitter
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.rows

// Splitting happens here rather than in readMessages so that callers narrowing the corpus
// (filterByCategory and friends) only pay for the messages they keep.
fun DataFrame<Message>.getSentences(splitter: SentenceSplitter = defaultSplitter): List<Sentence> {
    return this.rows().flatMap { row ->
        splitter.split(row.msg ?: "").map { sentence -> Sentence(row.msgId, row.date, sentence) }
    }
}

fun List<Sentence>.rawStrings(): List<String> {
    return this.map { it.sentence }
}
