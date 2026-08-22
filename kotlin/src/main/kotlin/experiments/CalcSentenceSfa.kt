package experiments

import algorithms.calcPca
import algorithms.calcSfa
import algorithms.writeFeatures
import algorithms.writeLoadings
import data.getSentences
import data.readMessages
import kotlin.io.path.Path

val sfaFeaturesPath = Path("data/sentence_sfa_features.csv")
val sfaLoadingsPath = Path("data/sentence_sfa_loadings.csv")

fun main() {
    val pca = readMessages()
        .filterByLindasPracticeLogs()
        .getSentences()
        .calcPca()

    val result = pca.calcSfa()

    val messages = result.messages
    println("Messages: ${messages.size} (${messages.first().date} -> ${messages.last().date})")
    println("Slowness: " + result.slowness.joinToString { "%.4f".format(it) })
    // Sanity check that SFA actually bought slowness over PCA: compare how strongly each series
    // correlates with itself one message later.
    for (j in 0 until result.nSlowFeatures) {
        println(
            "sf${j + 1} lag-1 autocorrelation %+.3f   pc${j + 1} %+.3f".format(
                lag1Autocorrelation(messages.map { it.slow[j] }),
                lag1Autocorrelation(messages.map { it.pca[j] }),
            )
        )
    }

    result.writeLoadings(sfaLoadingsPath)
    result.writeFeatures(sfaFeaturesPath)
}

private fun lag1Autocorrelation(series: List<Double>): Double {
    val mean = series.average()
    val variance = series.sumOf { (it - mean) * (it - mean) }
    if (variance == 0.0) return Double.NaN
    val covariance = (0 until series.size - 1).sumOf { (series[it] - mean) * (series[it + 1] - mean) }
    return covariance / variance
}
