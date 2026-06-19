package data

import org.jetbrains.kotlinx.dataframe.annotations.ColumnName
import org.jetbrains.kotlinx.dataframe.annotations.DataSchema

@DataSchema
interface RawMessage {
    @ColumnName("msg_id")
    val msgId: Long
    @ColumnName("thread_id")
    val threadId: Long
    val date: String
    @ColumnName("is_first_in_thread")
    val isFirstInThread: Boolean
    val category: String
    val author: String
    val title: String
    val msg: String?
}
