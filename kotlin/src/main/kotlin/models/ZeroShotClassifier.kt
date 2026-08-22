package models

interface ZeroShotClassifier : AutoCloseable {
    // Max total chars (batchSize * sentenceLength) per scoreBatch call; keeps GPU activation
    // memory roughly constant regardless of sentence length. Varies by model, so it lives here
    // rather than as a global constant.
    val batchSizeTokenBudget: Int

    fun scoreSingle(text: String, label: String): Float = scoreBatch(listOf(text), label).first()
    fun scoreBatch(texts: List<String>, label: String): List<Float>

    // Scores all texts for a label, internally batching by batchSizeTokenBudget so callers don't
    // have to. Texts are sorted by length before batching (tighter packing per batch) and results
    // are returned in the original input order. onBatch is invoked with each batch's size after
    // it's scored, e.g. to drive a progress bar.
    fun score(texts: List<String>, label: String, onBatch: (batchSize: Int) -> Unit = {}): List<Float> {
        if (texts.isEmpty()) return emptyList()

        val order = texts.indices.sortedBy { texts[it].length }
        val batches = createBatches(order.map { texts[it] }, batchSizeTokenBudget)

        val scores = FloatArray(texts.size)
        var offset = 0
        for (batch in batches) {
            val batchScores = scoreBatch(batch, label)
            for (i in batch.indices) {
                scores[order[offset + i]] = batchScores[i]
            }
            offset += batch.size
            onBatch(batch.size)
        }
        return scores.toList()
    }
}

// Greedily groups already-length-sorted texts so that batchSize * maxLength stays under
// maxBudget, keeping the per-batch GPU activation memory roughly constant regardless of text length.
private fun createBatches(sortedTexts: List<String>, maxBudget: Int): List<List<String>> {
    val batches = mutableListOf<List<String>>()
    var batch = mutableListOf<String>()
    for (text in sortedTexts) {
        if (batch.isNotEmpty() && (batch.size + 1) * text.length > maxBudget) {
            batches.add(batch)
            batch = mutableListOf()
        }
        batch.add(text)
    }
    if (batch.isNotEmpty()) batches.add(batch)
    return batches
}
