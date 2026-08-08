package experiments

import data.Sentence
import data.rawStrings
import me.tongfei.progressbar.ProgressBar
import models.ZeroShotClassifier
import models.defaultModel
import org.jetbrains.kotlinx.dataframe.AnyFrame
import org.jetbrains.kotlinx.dataframe.api.dataFrameOf
import org.jetbrains.kotlinx.dataframe.api.toColumn
import org.jetbrains.kotlinx.dataframe.io.writeCsv
import smile.feature.extraction.PCA
import java.nio.file.Path
import kotlin.io.path.createParentDirectories
import kotlin.math.abs

const val N_COMPONENTS = 5

// Holds everything a PCA fit over sentence label-scores produces, so downstream code can
// score new sentences into the same component space or build loadings/features output
// without re-fitting.
class PcaResult(
    val labels: List<String>,
    val nComponents: Int,
    val pca: PCA,
    // +1/-1 per component; fixComponentSigns flips `loadings`/`projected` post-hoc without
    // touching `pca`'s internal projection matrix, so any raw pca.getProjection(...).apply(x)
    // (e.g. in scoreSentences) must reapply these to stay in the same orientation.
    val signs: DoubleArray,
    val loadings: Array<DoubleArray>,
    val sentences: List<Sentence>,
    val projected: Array<DoubleArray>,
)

fun List<Sentence>.calcPca(
    labels: List<String> = experiments.labels,
    nComponents: Int = N_COMPONENTS,
    model: ZeroShotClassifier = defaultModel,
): PcaResult {
    val (x, rowIndexByText) = scoreDistinct(this.rawStrings(), labels, model)

    // No standardization: all features are entailment probabilities on the same [0,1] scale,
    // so raw variance differences across labels are exactly the signal we want PCA to surface.
    val pca = PCA.fit(x)
    val projected = pca.getProjection(nComponents).apply(x)
    val loadings = pca.loadings().transpose().toArray()
    val signs = fixComponentSigns(loadings, projected, nComponents)

    val expandedProjected = expandRows(this.map { it.sentence }, projected, rowIndexByText)

    return PcaResult(
        labels = labels,
        nComponents = nComponents,
        pca = pca,
        signs = signs,
        loadings = loadings,
        sentences = this,
        projected = expandedProjected,
    )
}

fun PcaResult.scoreSentences(texts: List<String>, model: ZeroShotClassifier = defaultModel): Array<DoubleArray> {
    val (x, rowIndexByText) = scoreDistinct(texts, labels, model)
    val projected = pca.getProjection(nComponents).apply(x)
    applySigns(projected, signs)

    return expandRows(texts, projected, rowIndexByText)
}

// Expands a deduped-by-text score matrix back to one row per original (possibly
// duplicate-containing) key, in the original order.
private fun expandRows(keys: List<String>, rows: Array<DoubleArray>, rowIndexByText: Map<String, Int>): Array<DoubleArray> =
    Array(keys.size) { i -> rows[rowIndexByText[keys[i]]!!] }

// Dedupe by text: identical texts would otherwise collide and pay for redundant model
// inference on every repeat, and (for calcPca's fit) would bias PCA's variance estimate
// towards repeated rows. Returns the deduped score matrix alongside the index each text's
// row lives at, so callers can look up any original (possibly duplicate-containing) text.
private fun scoreDistinct(texts: List<String>, labels: List<String>, model: ZeroShotClassifier): Pair<Array<DoubleArray>, Map<String, Int>> {
    val distinctTexts = texts.distinct()
    val rowIndexByText = distinctTexts.withIndex().associate { (i, s) -> s to i }
    val x = buildScoreMatrix(distinctTexts, labels, model)
    return x to rowIndexByText
}

private fun buildScoreMatrix(texts: List<String>, labels: List<String>, model: ZeroShotClassifier): Array<DoubleArray> {
    val x = Array(texts.size) { DoubleArray(labels.size) }
    ProgressBar("Scoring", labels.size.toLong() * texts.size).use { progressBar ->
        for ((labelIdx, label) in labels.withIndex()) {
            val scores = model.score(texts, label) { n -> progressBar.stepBy(n.toLong()) }
            for (i in scores.indices) x[i][labelIdx] = scores[i].toDouble()
        }
    }
    return x
}

// SVD components have an arbitrary sign; pin each PC so its largest-magnitude loading is positive,
// flipping the matching projected column to match, so re-fits after label/cache changes stay comparable.
// Returns the sign applied per component so callers can reapply it to out-of-sample projections.
private fun fixComponentSigns(loadings: Array<DoubleArray>, projected: Array<DoubleArray>, nComponents: Int): DoubleArray {
    val signs = DoubleArray(nComponents) { 1.0 }
    for (j in 0 until nComponents) {
        val maxIdx = loadings[j].indices.maxBy { abs(loadings[j][it]) }
        if (loadings[j][maxIdx] < 0) {
            signs[j] = -1.0
            for (i in loadings[j].indices) loadings[j][i] = -loadings[j][i]
        }
    }
    applySigns(projected, signs)
    return signs
}

private fun applySigns(projected: Array<DoubleArray>, signs: DoubleArray) {
    for (row in projected) {
        for (j in signs.indices) row[j] *= signs[j]
    }
}

private fun buildPcColumns(nComponents: Int, componentValues: (Int) -> List<Double>) =
    (0 until nComponents).map { j -> componentValues(j).toColumn("pc${j + 1}") }

fun PcaResult.loadingsDataFrame(): AnyFrame {
    val labelColumn = labels.toColumn("label")
    val pcColumns = buildPcColumns(nComponents) { j -> loadings[j].toList() }
    return dataFrameOf(listOf(labelColumn) + pcColumns)
}

fun PcaResult.featuresDataFrame(): AnyFrame {
    val idColumns = listOf(
        sentences.map { it.msgId }.toColumn("msgId"),
        sentences.map { it.date }.toColumn("date"),
        sentences.map { it.sentence }.toColumn("sentence"),
    )
    val pcColumns = buildPcColumns(nComponents) { j -> projected.map { it[j] } }
    return dataFrameOf(idColumns + pcColumns)
}

fun PcaResult.writeLoadings(path: Path) {
    path.createParentDirectories()
    loadingsDataFrame().writeCsv(path.toString())
}

fun PcaResult.writeFeatures(path: Path) {
    path.createParentDirectories()
    featuresDataFrame().writeCsv(path.toString())
}
