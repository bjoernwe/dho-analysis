package data

import org.jetbrains.kotlinx.dataframe.annotations.DataSchema

@DataSchema
interface Message: RawMessage {
    val threadAuthor: String
    val sentences: List<String>
}
