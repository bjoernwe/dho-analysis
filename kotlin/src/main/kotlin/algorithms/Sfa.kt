package algorithms

import smile.math.MathEx
import smile.math.matrix.Matrix
import kotlin.math.sqrt

// Eigenvalues below this fraction of the largest one are treated as noise directions and
// dropped, so whitening never divides by an (almost) zero variance.
private const val EIGENVALUE_TOLERANCE = 1e-10

// A fitted Slow Feature Analysis: the affine map from the input space onto the directions whose
// time derivative varies least. Knows nothing about what the series represents.
class Sfa(
    // Column means of the series this was fitted on; subtracted before projecting.
    val mean: DoubleArray,
    // [slow feature][input dim], with the whitening matrix already folded in, so `transform` is a
    // single matrix-vector product and callers composing SFA with an upstream linear stage (PCA,
    // say) can multiply straight through it.
    val projection: Array<DoubleArray>,
    // Mean squared time derivative of each slow feature, ascending. Smaller means slower.
    val slowness: DoubleArray,
) {
    fun transform(x: DoubleArray): DoubleArray =
        DoubleArray(projection.size) { m ->
            x.indices.sumOf { i -> projection[m][i] * (x[i] - mean[i]) }
        }

    fun transform(series: Array<DoubleArray>): Array<DoubleArray> =
        Array(series.size) { transform(series[it]) }
}

// Fits SFA on an ordered series: the directions whose time derivative has the smallest variance,
// subject to the outputs being zero-mean, unit-variance and mutually decorrelated.
//
// The whitening step is what makes those constraints hold and is not optional. SFA is really the
// generalised eigenproblem `Cd b = lambda C b`; sphering the input reduces it to the ordinary
// symmetric eigenproblem below. Skipping it would just surface the lowest-variance input direction
// instead of the slowest one, since shrinking a direction shrinks its derivative along with it.
//
// `transitions` lists the indices t for which the step t -> t+1 counts as one time step, so callers
// whose series has gaps or hard boundaries can leave those pairs out. Defaults to every pair.
fun fitSfa(
    series: Array<DoubleArray>,
    nSlowFeatures: Int,
    transitions: List<Int> = (0 until series.size - 1).toList(),
): Sfa {
    require(series.size >= 2) { "SFA needs at least two time steps, got ${series.size}" }
    require(transitions.isNotEmpty()) { "SFA needs at least one transition" }
    val dim = series[0].size
    require(nSlowFeatures <= dim) { "Cannot extract $nSlowFeatures slow features from $dim dimensions" }

    val mean = MathEx.colMeans(series)
    val centered = Array(series.size) { i -> DoubleArray(dim) { j -> series[i][j] - mean[j] } }

    val whitening = whiteningMatrix(covariance(centered))
    require(whitening.size == dim) {
        "Series is rank-deficient (${whitening.size} of $dim usable directions); reduce the input dimension"
    }
    val whitened = Array(centered.size) { i -> multiply(whitening, centered[i]) }

    val derivatives = transitions.map { t ->
        DoubleArray(dim) { j -> whitened[t + 1][j] - whitened[t][j] }
    }.toTypedArray()

    // Derivatives are differences, so their mean is a boundary effect rather than an offset to
    // remove; the second moment is the quantity SFA minimises.
    val (eigenvalues, eigenvectors) = symmetricEigenAscending(secondMoment(derivatives))

    val projection = Array(nSlowFeatures) { m ->
        DoubleArray(dim) { i -> (0 until dim).sumOf { k -> eigenvectors[m][k] * whitening[k][i] } }
    }
    return Sfa(mean, projection, eigenvalues.copyOf(nSlowFeatures))
}

// Covariance of already-centered rows.
internal fun covariance(centered: Array<DoubleArray>): Array<DoubleArray> =
    gram(centered, (centered.size - 1).toDouble())

// Second moment about zero, for data whose mean is not subtracted.
private fun secondMoment(x: Array<DoubleArray>): Array<DoubleArray> = gram(x, x.size.toDouble())

private fun gram(x: Array<DoubleArray>, divisor: Double): Array<DoubleArray> {
    val dim = x[0].size
    val out = Array(dim) { DoubleArray(dim) }
    for (row in x) {
        for (i in 0 until dim) {
            for (j in i until dim) out[i][j] += row[i] * row[j]
        }
    }
    // Fill the lower triangle by mirroring so the result is exactly symmetric, which the
    // eigensolver relies on.
    for (i in 0 until dim) {
        for (j in i until dim) {
            out[i][j] /= divisor
            out[j][i] = out[i][j]
        }
    }
    return out
}

// Rows of the returned matrix map a centered vector onto unit-variance, decorrelated coordinates.
// Directions whose eigenvalue is negligible relative to the largest are dropped rather than
// amplified, so the row count may be smaller than the input dimension.
internal fun whiteningMatrix(covariance: Array<DoubleArray>): Array<DoubleArray> {
    val (eigenvalues, eigenvectors) = symmetricEigenAscending(covariance)
    val threshold = eigenvalues.last() * EIGENVALUE_TOLERANCE
    return eigenvalues.indices
        .filter { eigenvalues[it] > threshold }
        .map { k ->
            val scale = 1.0 / sqrt(eigenvalues[k])
            DoubleArray(eigenvectors[k].size) { i -> eigenvectors[k][i] * scale }
        }
        .toTypedArray()
}

// Eigendecomposition of a symmetric matrix, ascending by eigenvalue, with the eigenvectors returned
// as rows. LAPACK's symmetric path already returns ascending eigenvalues; the explicit sort keeps
// that guarantee independent of which path Matrix.eigen picks.
private fun symmetricEigenAscending(matrix: Array<DoubleArray>): Pair<DoubleArray, Array<DoubleArray>> {
    val evd = Matrix.of(matrix).eigen(false, true, false)
    val order = evd.wr.indices.sortedBy { evd.wr[it] }
    val values = DoubleArray(order.size) { evd.wr[order[it]] }
    val vectors = Array(order.size) { k ->
        DoubleArray(matrix.size) { i -> evd.Vr.get(i, order[k]) }
    }
    return values to vectors
}

private fun multiply(matrix: Array<DoubleArray>, vector: DoubleArray): DoubleArray =
    DoubleArray(matrix.size) { k -> vector.indices.sumOf { i -> matrix[k][i] * vector[i] } }
