package experiments

import data.readMessages
import me.tongfei.progressbar.ProgressBar
import models.defaultModel
import org.jetbrains.kotlinx.dataframe.DataColumn
import org.jetbrains.kotlinx.dataframe.api.toDataFrame
import org.jetbrains.kotlinx.dataframe.io.writeCsv
import smile.feature.extraction.PCA
import java.io.File
import kotlin.io.path.Path
import kotlin.io.path.createParentDirectories
import kotlin.math.abs

const val N_COMPONENTS = 3
val featuresPath = Path("cache/sentence_features.csv")
val loadingsPath = Path("cache/pca_loadings.csv")

data class SentenceRef(val msgId: Long, val date: String, val sentence: String)

data class SentenceFeatures(
    val msgId: Long,
    val date: String,
    val sentence: String,
    val pc1: Double,
    val pc2: Double,
    val pc3: Double,
)

fun main() {
    val sentenceRefs = readSentenceRefs().take(10_000)

    val sentences = sentenceRefs.map { it.sentence }.sortedBy { it.length }
    val rowIndexByText = sentences.withIndex().associate { (i, s) -> s to i }

    val x = buildScoreMatrix(sentences, rowIndexByText)

    // No standardization: all features are entailment probabilities on the same [0,1] scale,
    // so raw variance differences across labels are exactly the signal we want PCA to surface.
    val pca = PCA.fit(x)
    val projected = pca.getProjection(N_COMPONENTS).apply(x)
    val loadings = pca.loadings().transpose().toArray()
    fixComponentSigns(loadings, projected)

    writeLoadings(loadings)
    writeFeatures(sentenceRefs, rowIndexByText, projected)
}

private fun readSentenceRefs(): List<SentenceRef> {
    val messages = readMessages()
    // Cross-file access to typed row properties on this DataFrame still doesn't resolve with the
    // current DataFrame codegen, so go through untyped column access instead.
    val msgIds = (messages["msg_id"] as DataColumn<Long>).toList()
    val dates = (messages["date"] as DataColumn<String>).toList()
    val sentenceLists = (messages["sentences"] as DataColumn<List<String>>).toList()
    return msgIds.indices.flatMap { i ->
        sentenceLists[i].map { sentence -> SentenceRef(msgIds[i], dates[i], sentence) }
    }
}

private fun buildScoreMatrix(sentences: List<String>, rowIndexByText: Map<String, Int>): Array<DoubleArray> {
    val x = Array(sentences.size) { DoubleArray(labels.size) }
    defaultModel.use { model ->
        ProgressBar("Scoring", labels.size.toLong() * sentences.size).use { progressBar ->
            for ((labelIdx, label) in labels.withIndex()) {
                for (batch in createBatches(sentences)) {
                    val scores = model.scoreBatch(batch, label)
                    for ((offset, sentence) in batch.withIndex()) {
                        x[rowIndexByText[sentence]!!][labelIdx] = scores[offset].toDouble()
                    }
                    progressBar.stepBy(batch.size.toLong())
                }
            }
        }
    }
    return x
}

// SVD components have an arbitrary sign; pin each PC so its largest-magnitude loading is positive,
// flipping the matching projected column to match, so re-fits after label/cache changes stay comparable.
private fun fixComponentSigns(loadings: Array<DoubleArray>, projected: Array<DoubleArray>) {
    for (j in 0 until N_COMPONENTS) {
        val maxIdx = loadings[j].indices.maxBy { abs(loadings[j][it]) }
        if (loadings[j][maxIdx] < 0) {
            for (i in loadings[j].indices) loadings[j][i] = -loadings[j][i]
            for (row in projected) row[j] = -row[j]
        }
    }
}

private fun writeLoadings(loadings: Array<DoubleArray>) {
    loadingsPath.createParentDirectories()
    File(loadingsPath.toString()).bufferedWriter().use { writer ->
        writer.write("label," + (1..N_COMPONENTS).joinToString(",") { "pc$it" })
        writer.newLine()
        for ((i, label) in labels.withIndex()) {
            writer.write(label + "," + (0 until N_COMPONENTS).joinToString(",") { j -> loadings[j][i].toString() })
            writer.newLine()
        }
    }
}

private fun writeFeatures(
    sentenceRefs: List<SentenceRef>,
    rowIndexByText: Map<String, Int>,
    projected: Array<DoubleArray>,
) {
    val sentenceFeatures = sentenceRefs.map { ref ->
        val row = projected[rowIndexByText[ref.sentence]!!]
        SentenceFeatures(ref.msgId, ref.date, ref.sentence, row[0], row[1], row[2])
    }
    featuresPath.createParentDirectories()
    sentenceFeatures.toDataFrame().writeCsv(featuresPath.toString())
}
