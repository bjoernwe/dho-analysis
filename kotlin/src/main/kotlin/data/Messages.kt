package data

import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.convertTo
import org.jetbrains.kotlinx.dataframe.io.readJsonStr
import java.io.File

fun readMessages(path: String = "data/messages.jsonl"): DataFrame<Message> {
    val json = File(path).useLines { lines ->
        lines.filter { it.isNotBlank() }.joinToString(",", prefix = "[", postfix = "]")
    }
    return DataFrame.readJsonStr(json).convertTo<Message>()
}
