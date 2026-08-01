package data

fun readRawSentences(): List<String> {
    val messages = readMessages()
    val sentences = messages.sentences.toList().flatten().sortedBy { it.length }
    return sentences
}

// Typed row properties only resolve in the same file/function as the convertTo<Message> call
// that produces this frame, so this lives here rather than in each consumer.
fun readSentences(): List<Sentence> {
    val messages = readMessages()
    val msgIds = messages.msgId.toList()
    val dates = messages.date.toList()
    val sentenceLists = messages.sentences.toList()
    return msgIds.indices.flatMap { i ->
        sentenceLists[i].map { sentence -> Sentence(msgIds[i], dates[i], sentence) }
    }
}
