package experiments

const val BATCH_SIZE_TOKEN_BUDGET = 47_000


// Greedily groups already-length-sorted sentences so that batchSize * maxLength stays under
// maxBudget, keeping the per-batch GPU activation memory roughly constant regardless of sentence length.
fun createBatches(sortedSentences: List<String>, maxBudget: Int = BATCH_SIZE_TOKEN_BUDGET): List<List<String>> {
    val batches = mutableListOf<List<String>>()
    var batch = mutableListOf<String>()
    for (sentence in sortedSentences) {
        if (batch.isNotEmpty() && (batch.size + 1) * sentence.length > maxBudget) {
            batches.add(batch)
            batch = mutableListOf()
        }
        batch.add(sentence)
    }
    if (batch.isNotEmpty()) batches.add(batch)
    return batches
}
