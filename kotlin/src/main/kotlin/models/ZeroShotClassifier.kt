package models

interface ZeroShotClassifier : AutoCloseable {
    fun score(text: String, label: String): Float = scoreBatch(listOf(text), label).first()
    fun scoreBatch(texts: List<String>, label: String): List<Float>
}
