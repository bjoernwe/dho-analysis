package algorithms

import data.Sentence
import data.rawStrings
import me.tongfei.progressbar.ProgressBar
import models.ZeroShotClassifier
import models.defaultModel
import org.jetbrains.kotlinx.dataframe.AnyFrame
import org.jetbrains.kotlinx.dataframe.api.dataFrameOf
import org.jetbrains.kotlinx.dataframe.api.toColumn
import org.jetbrains.kotlinx.dataframe.io.writeCsv
import sfa.Pca
import sfa.fitPca
import java.nio.file.Path
import kotlin.io.path.createParentDirectories

const val N_COMPONENTS = 5

// Holds everything a PCA fit over sentence label-scores produces, so downstream code can
// score new sentences into the same component space or build loadings/features output
// without re-fitting.
class PcaResult(
    val labels: List<String>,
    val nComponents: Int,
    val pca: Pca,
    val sentences: List<Sentence>,
    val projected: Array<DoubleArray>,
) {
    // `center` plus the sign-pinned `loadings` are the fit's canonical form: every projection goes
    // through them, so there is no second, un-oriented representation for callers to reconcile.
    val center: DoubleArray get() = pca.center
    val loadings: Array<DoubleArray> get() = pca.loadings
}

fun List<Sentence>.calcPca(
    labels: List<String> = experiments.labels,
    nComponents: Int = N_COMPONENTS,
    model: ZeroShotClassifier = defaultModel,
): PcaResult {
    val texts = rawStrings()
    val (x, rowIndexByText) = scoreDistinct(texts, labels, model)

    // No standardization: all features are entailment probabilities on the same [0,1] scale,
    // so raw variance differences across labels are exactly the signal we want PCA to surface.
    val pca = fitPca(x, nComponents)

    val projected = pca.transform(x)

    return PcaResult(
        labels = labels,
        nComponents = nComponents,
        pca = pca,
        sentences = this,
        projected = expandRows(texts, projected, rowIndexByText),
    )
}

// Projects a raw label-score vector onto the first `nComponents` sign-pinned components.
fun PcaResult.project(x: DoubleArray): DoubleArray = pca.transform(x)

fun PcaResult.scoreSentences(texts: List<String>, model: ZeroShotClassifier = defaultModel): Array<DoubleArray> {
    val (x, rowIndexByText) = scoreDistinct(texts, labels, model)
    return expandRows(texts, Array(x.size) { project(x[it]) }, rowIndexByText)
}

// Expands a deduped-by-text score matrix back to one row per original (possibly
// duplicate-containing) key, in the original order.
internal fun expandRows(keys: List<String>, rows: Array<DoubleArray>, rowIndexByText: Map<String, Int>): Array<DoubleArray> =
    Array(keys.size) { i -> rows[rowIndexByText[keys[i]]!!] }

// Dedupe by text: identical texts would otherwise collide and pay for redundant model
// inference on every repeat, and (for calcPca's fit) would bias PCA's variance estimate
// towards repeated rows. Returns the deduped score matrix alongside the index each text's
// row lives at, so callers can look up any original (possibly duplicate-containing) text.
internal fun scoreDistinct(texts: List<String>, labels: List<String>, model: ZeroShotClassifier): Pair<Array<DoubleArray>, Map<String, Int>> {
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

// Builds `label` plus one column per component, e.g. pc1..pc5 or sf1..sf5.
internal fun loadingsFrame(labels: List<String>, loadings: Array<DoubleArray>, n: Int, prefix: String): AnyFrame =
    dataFrameOf(listOf(labels.toColumn("label")) + componentColumns(n, prefix) { j -> loadings[j].toList() })

// Builds the sentence id columns plus one column per component.
internal fun featuresFrame(sentences: List<Sentence>, projected: Array<DoubleArray>, n: Int, prefix: String): AnyFrame =
    dataFrameOf(
        listOf(
            sentences.map { it.msgId }.toColumn("msgId"),
            sentences.map { it.date }.toColumn("date"),
            sentences.map { it.sentence }.toColumn("sentence"),
        ) + componentColumns(n, prefix) { j -> projected.map { it[j] } }
    )

internal fun AnyFrame.writeCsvTo(path: Path) {
    path.createParentDirectories()
    writeCsv(path.toString())
}

private fun componentColumns(n: Int, prefix: String, values: (Int) -> List<Double>) =
    (0 until n).map { j -> values(j).toColumn("$prefix${j + 1}") }

fun PcaResult.loadingsDataFrame(): AnyFrame = loadingsFrame(labels, loadings, nComponents, "pc")

fun PcaResult.featuresDataFrame(): AnyFrame = featuresFrame(sentences, projected, nComponents, "pc")

fun PcaResult.writeLoadings(path: Path) = loadingsDataFrame().writeCsvTo(path)

fun PcaResult.writeFeatures(path: Path) = featuresDataFrame().writeCsvTo(path)
