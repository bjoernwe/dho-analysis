package data

fun readRawSentences(): List<String> {
    return readSentences().map { it.sentence }
}

fun readSentences(): List<Sentence> {
    val messages = readMessages()
    return messages.sentences.toList().flatten()
}
