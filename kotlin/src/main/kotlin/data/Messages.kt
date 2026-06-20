package data

import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.convertTo
import org.jetbrains.kotlinx.dataframe.api.fill
import org.jetbrains.kotlinx.dataframe.api.filter
import org.jetbrains.kotlinx.dataframe.api.map
import org.jetbrains.kotlinx.dataframe.api.with
import org.jetbrains.kotlinx.dataframe.io.readJsonStr
import models.SentenceSplitter
import java.io.File

fun readMessages(splitter: SentenceSplitter, path: String = "data/messages.jsonl"): DataFrame<Message> {
    val rawMessages = readRawMessages(path)
    val threadAuthors = rawMessages.filter { it.isFirstInThread }.map { it.threadId to it.author }.toMap()
    val json = readLinesAsJsonArray(path)
    val messages = DataFrame.readJsonStr(json).convertTo<Message> { fill { threadAuthor }.with { threadAuthors[threadId] ?: "n/a" }; fill { sentences }.with { splitter.split(msg ?: "") } }
    return messages
}

private fun readRawMessages(path: String): DataFrame<RawMessage> {
    val json = readLinesAsJsonArray(path)
    return DataFrame.readJsonStr(json).convertTo<RawMessage>()
}

private fun readLinesAsJsonArray(path: String): String {
    return File(path).useLines { lines ->
        lines.filter { it.isNotBlank() }.joinToString(",", prefix = "[", postfix = "]")
    }
}
