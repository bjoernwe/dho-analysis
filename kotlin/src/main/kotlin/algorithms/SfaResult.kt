package algorithms

import data.Sentence
import models.ZeroShotClassifier
import models.defaultModel
import org.jetbrains.kotlinx.dataframe.AnyFrame
import sfa.Sfa
import sfa.fitSfa
import sfa.pinSigns
import smile.math.MathEx
import java.nio.file.Path
import java.time.Duration
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

const val N_SLOW_FEATURES = 5

// Sentence `date` is always "yyyy-MM-dd HH:mm:ss", so lexicographic ordering is already
// chronological; parsing is only needed to measure gaps in real time units.
private val DATE_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")

// One step of the message-level series SFA was fitted on, in chronological order: the message's
// mean PCA scores and the slow features derived from them. Kept for slowness diagnostics.
class MessageStep(val date: String, val pca: DoubleArray, val slow: DoubleArray)

// Holds a fitted SFA on top of a PCA fit. `loadings` is the composed PCA -> SFA map expressed
// back in raw label space, so it can be plotted like PCA loadings and used to score sentences
// in one step.
class SfaResult(
    val labels: List<String>,
    // [slow feature][label], composed over the whole chain (centering, PCA, whitening, SFA).
    val loadings: Array<DoubleArray>,
    val offsets: DoubleArray,
    // Ascending mean squared time derivative of each slow feature. Smaller means slower.
    val slowness: DoubleArray,
    val sentences: List<Sentence>,
    val projected: Array<DoubleArray>,
    val messages: List<MessageStep>,
) {
    val nSlowFeatures: Int get() = loadings.size
}

// Runs SFA on the message-level average of this PCA fit's component scores.
//
// Sentences carry only their message's timestamp, so consecutive sentences within a message have
// no time difference at all; averaging per message turns the corpus into a well-defined time
// series with one step per post. Messages are sorted by date because the source jsonl is grouped
// by thread, not chronological.
//
// `maxGapDays` optionally drops transitions that span more than the given number of days, for
// cases where a long silence should not count as a single time step.
fun PcaResult.calcSfa(
    nSlowFeatures: Int = N_SLOW_FEATURES,
    maxGapDays: Double? = null,
): SfaResult {
    val messages = aggregateByMessage(sentences, projected).sortedBy { it.date }
    val series = Array(messages.size) { messages[it].scores }
    val sfa = fitSfa(series, nSlowFeatures, transitions(messages, maxGapDays))

    // Compose the whole chain down to raw label scores. PcaResult maps a label vector x to
    // p[j] = sum_l loadings[j][l] * (x[l] - center[l]); feeding that into SFA's
    // y[m] = sum_j projection[m][j] * (p[j] - mean[j]) and collecting terms gives a single affine
    // map from x straight to the slow features.
    val composed = Array(nSlowFeatures) { m ->
        DoubleArray(labels.size) { l ->
            (0 until nComponents).sumOf { j -> sfa.projection[m][j] * loadings[j][l] }
        }
    }
    val offsets = DoubleArray(nSlowFeatures) { m ->
        -composed[m].indices.sumOf { l -> composed[m][l] * center[l] } -
            sfa.projection[m].indices.sumOf { j -> sfa.projection[m][j] * sfa.mean[j] }
    }

    // Slow features are defined only up to sign; pin each one the same way PCA components are
    // pinned, so re-fits stay comparable. Folding the flip straight back into the projection keeps
    // one oriented map -- every projection below then comes out correct with no post-hoc
    // correction pass for a future caller to forget.
    val signs = pinSigns(composed, nSlowFeatures)
    val oriented = Sfa(
        mean = sfa.mean,
        projection = Array(nSlowFeatures) { m -> DoubleArray(nComponents) { j -> signs[m] * sfa.projection[m][j] } },
        slowness = sfa.slowness,
    )
    for (m in signs.indices) offsets[m] *= signs[m]

    // Sentence-level slow features come from the sentence's PCA scores rather than a re-scoring
    // pass: p is exactly what the composed loadings would reconstruct from the label scores, and
    // SfaTest asserts the two paths agree.
    val messageSlow = oriented.transform(series)

    return SfaResult(
        labels = labels,
        loadings = composed,
        offsets = offsets,
        slowness = sfa.slowness,
        sentences = sentences,
        projected = oriented.transform(projected),
        messages = messages.indices.map { MessageStep(messages[it].date, series[it], messageSlow[it]) },
    )
}

fun SfaResult.scoreSentences(texts: List<String>, model: ZeroShotClassifier = defaultModel): Array<DoubleArray> {
    val (x, rowIndexByText) = scoreDistinct(texts, labels, model)
    return expandRows(texts, Array(x.size) { scoreLabelVector(x[it]) }, rowIndexByText)
}

// Applies the composed PCA -> SFA map to a single raw label-score vector.
fun SfaResult.scoreLabelVector(x: DoubleArray): DoubleArray =
    DoubleArray(nSlowFeatures) { m ->
        x.indices.sumOf { l -> loadings[m][l] * x[l] } + offsets[m]
    }

private class MessageScores(val date: String, val scores: DoubleArray)

private fun aggregateByMessage(sentences: List<Sentence>, projected: Array<DoubleArray>): List<MessageScores> =
    sentences.indices.groupBy { sentences[it].msgId }.map { (_, rows) ->
        MessageScores(sentences[rows.first()].date, MathEx.colMeans(Array(rows.size) { projected[rows[it]] }))
    }

// Indices t for which the step t -> t+1 counts as one time step.
private fun transitions(messages: List<MessageScores>, maxGapDays: Double?): List<Int> {
    val all = (0 until messages.size - 1).toList()
    if (maxGapDays == null) return all
    val instants = messages.map { LocalDateTime.parse(it.date, DATE_FORMAT) }
    return all.filter { t ->
        Duration.between(instants[t], instants[t + 1]).seconds / 86400.0 <= maxGapDays
    }
}

fun SfaResult.loadingsDataFrame(): AnyFrame =
    loadingsFrame(labels, loadings, nSlowFeatures, "sf")

fun SfaResult.featuresDataFrame(): AnyFrame =
    featuresFrame(sentences, projected, nSlowFeatures, "sf")

fun SfaResult.writeLoadings(path: Path) = loadingsDataFrame().writeCsvTo(path)

fun SfaResult.writeFeatures(path: Path) = featuresDataFrame().writeCsvTo(path)
