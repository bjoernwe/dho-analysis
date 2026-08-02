package data

import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.convertTo
import org.jetbrains.kotlinx.dataframe.api.fill
import org.jetbrains.kotlinx.dataframe.api.filter
import org.jetbrains.kotlinx.dataframe.api.map
import org.jetbrains.kotlinx.dataframe.api.with
import org.jetbrains.kotlinx.dataframe.io.readJsonStr
import models.SentenceSplitter
import models.defaultSplitter
import java.io.File

fun readMessages(path: String = "data/messages.jsonl", splitter: SentenceSplitter = defaultSplitter): DataFrame<Message> {
    val json = readLinesAsJsonArray(path)
    val rawMessages = DataFrame.readJsonStr(json).convertTo<RawMessage>()
    val threadAuthors = rawMessages.filter { it.isFirstInThread }.map { it.threadId to it.author }.toMap()
    return DataFrame.readJsonStr(json).convertTo<Message> {
        fill { threadAuthor }.with { threadAuthors[threadId] ?: "n/a" }
        fill { sentences }.with {
            splitter.split(msg ?: "").map { sentence -> Sentence(msgId, date, sentence) }
        }
    }
}

private fun readLinesAsJsonArray(path: String): String {
    return File(path).useLines { lines ->
        lines.filter { it.isNotBlank() }.joinToString(",", prefix = "[", postfix = "]")
    }
}

fun DataFrame<Message>.filterByCategory(category: String = "PracticeLogs"): DataFrame<Message> {
    return this.filter { it.category == category }
}

fun DataFrame<Message>.filterByAuthor(author: String = "Linda ”Polly Ester” Ö"): DataFrame<Message> {
    return this.filter { it.author == author }
}

fun DataFrame<Message>.filterByThreadAuthor(): DataFrame<Message> {
    return this.filter { it.threadAuthor == it.author }
}
