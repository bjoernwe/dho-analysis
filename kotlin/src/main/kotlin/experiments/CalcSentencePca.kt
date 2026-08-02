package experiments

import data.Sentence
import data.filterByAuthor
import data.filterByCategory
import data.filterByOc
import data.getSentences
import data.readMessages
import data.sortByDescendingLength
import me.tongfei.progressbar.ProgressBar
import models.defaultModel
import org.jetbrains.kotlinx.dataframe.api.dataFrameOf
import org.jetbrains.kotlinx.dataframe.api.toColumn
import org.jetbrains.kotlinx.dataframe.io.writeCsv
import smile.feature.extraction.PCA
import kotlin.io.path.Path
import kotlin.io.path.createParentDirectories
import kotlin.math.abs

const val N_COMPONENTS = 5
val featuresPath = Path("data/sentence_features.csv")
val loadingsPath = Path("data/sentence_pca_loadings.csv")

fun main() {
    val sentenceRefs = readMessages()
        .filterByCategory("PracticeLogs")
        .filterByAuthor("Linda ”Polly Ester” Ö")
        .filterByOc()
        .getSentences()
        .sortByDescendingLength()

    // Dedupe by text: identical sentences would otherwise collide in rowIndexByText and pay for
    // redundant model inference on every repeat.
    val sentences = sentenceRefs.map { it.sentence }.distinct().sortedBy { it.length }
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

private fun buildScoreMatrix(sentences: List<String>, rowIndexByText: Map<String, Int>): Array<DoubleArray> {
    val x = Array(sentences.size) { DoubleArray(labels.size) }
    defaultModel.use { model ->
        ProgressBar("Scoring", labels.size.toLong() * sentences.size).use { progressBar ->
            for ((labelIdx, label) in labels.withIndex()) {
                val scores = model.score(sentences, label) { n -> progressBar.stepBy(n.toLong()) }
                for ((offset, sentence) in sentences.withIndex()) {
                    x[rowIndexByText[sentence]!!][labelIdx] = scores[offset].toDouble()
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

private fun buildPcColumns(componentValues: (Int) -> List<Double>) =
    (0 until N_COMPONENTS).map { j -> componentValues(j).toColumn("pc${j + 1}") }

private fun writeLoadings(loadings: Array<DoubleArray>) {
    val labelColumn = labels.toColumn("label")
    val pcColumns = buildPcColumns { j -> loadings[j].toList() }
    loadingsPath.createParentDirectories()
    dataFrameOf(listOf(labelColumn) + pcColumns).writeCsv(loadingsPath.toString())
}

private fun writeFeatures(
    sentenceRefs: List<Sentence>,
    rowIndexByText: Map<String, Int>,
    projected: Array<DoubleArray>,
) {
    val rows = sentenceRefs.map { ref -> projected[rowIndexByText[ref.sentence]!!] }
    val idColumns = listOf(
        sentenceRefs.map { it.msgId }.toColumn("msgId"),
        sentenceRefs.map { it.date }.toColumn("date"),
        sentenceRefs.map { it.sentence }.toColumn("sentence"),
    )
    val pcColumns = buildPcColumns { j -> rows.map { it[j] } }
    featuresPath.createParentDirectories()
    dataFrameOf(idColumns + pcColumns).writeCsv(featuresPath.toString())
}
