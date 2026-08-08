package models

interface SentenceSplitter {
    fun split(text: String): List<String>
}
