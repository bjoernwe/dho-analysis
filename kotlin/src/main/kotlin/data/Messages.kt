package data

import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.convertTo
import org.jetbrains.kotlinx.dataframe.api.fill
import org.jetbrains.kotlinx.dataframe.api.filter
import org.jetbrains.kotlinx.dataframe.api.map
import org.jetbrains.kotlinx.dataframe.api.with
import org.jetbrains.kotlinx.dataframe.io.readJsonStr
import java.io.File

fun readMessages(path: String = "data/messages.jsonl"): DataFrame<Message> {
    val json = readLinesAsJsonArray(path)
    val rawMessages = DataFrame.readJsonStr(json).convertTo<RawMessage>()
    val threadAuthors = rawMessages.filter { it.isFirstInThread }.map { it.threadId to it.author }.toMap()
    val messages = DataFrame.readJsonStr(json).convertTo<Message> { fill { threadAuthor }.with { threadAuthors[threadId] ?: "n/a" } }
    return messages
}

private fun readLinesAsJsonArray(path: String): String {
    return File(path).useLines { lines ->
        lines.filter { it.isNotBlank() }.joinToString(",", prefix = "[", postfix = "]")
    }
}
