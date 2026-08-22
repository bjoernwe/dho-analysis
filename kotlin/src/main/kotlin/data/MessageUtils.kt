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
    // Parsed once and reused: the thread-author lookup and the result frame are the same rows,
    // and re-parsing the corpus costs a full pass over tens of megabytes of JSON.
    val parsed = DataFrame.readJsonStr(readLinesAsJsonArray(path))
    val threadAuthors = parsed.convertTo<RawMessage>()
        .filter { it.isFirstInThread }
        .map { it.threadId to it.author }
        .toMap()
    return parsed.convertTo<Message> {
        fill { threadAuthor }.with { threadAuthors[threadId] ?: "n/a" }
    }
}

private fun readLinesAsJsonArray(path: String): String {
    return File(path).useLines { lines ->
        lines.filter { it.isNotBlank() }.joinToString(",", prefix = "[", postfix = "]")
    }
}

fun DataFrame<Message>.filterByCategory(category: String): DataFrame<Message> {
    return this.filter { it.category == category }
}

fun DataFrame<Message>.filterByAuthor(author: String): DataFrame<Message> {
    return this.filter { it.author == author }
}

fun DataFrame<Message>.filterByThreadAuthor(): DataFrame<Message> {
    return this.filter { it.threadAuthor == it.author }
}
